"""Real layer-streamed inference over OpenLLM shards.

The model skeleton is built on the ``meta`` device, so no weight is
resident between passes.  A forward pre-hook materialises exactly one
decoder layer from its shard right before that layer runs; the forward
hook releases it again.  Peak resident weights are therefore one layer
plus the embedding / head, which is the whole point of the project.

Buffers (rotary ``inv_freq`` and friends) are created for real at build
time via ``init_empty_weights(include_buffers=False)`` and are never
released, so the meta round trip cannot corrupt them.
"""

import json
import queue
import threading
from pathlib import Path

MANIFEST = "manifest.json"


class Event:
    __slots__ = ("type", "index", "text")

    def __init__(self, type, index=None, text=""):
        self.type = type
        self.index = index
        self.text = text

    def __repr__(self):
        return "Event(%r, %r)" % (self.type, self.index)


def find_layers(model):
    """Locate the decoder block list for the common HF architectures."""
    for path in (
        "model.layers",
        "model.decoder.layers",
        "transformer.h",
        "gpt_neox.layers",
        "layers",
    ):
        node = model
        for part in path.split("."):
            node = getattr(node, part, None)
            if node is None:
                break
        if node is not None and len(node):
            return list(node)
    return None


class LayerStreamer:
    def __init__(self, model_dir, device="cpu", dtype="auto", max_new_tokens=256):
        import torch

        self.torch = torch
        self.model_dir = Path(model_dir)
        self.manifest = json.loads((self.model_dir / MANIFEST).read_text(encoding="utf-8"))
        self.layer_count = int(self.manifest["layers"])
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.dtype = self._resolve_dtype(dtype)
        self.paths = self._shard_paths()
        self.model = None
        self.tokenizer = None
        self.config = None
        self._sink = None
        self._blocks = []

    # ------------------------------------------------------------------ setup

    def _resolve_dtype(self, dtype):
        torch = self.torch
        table = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if dtype in (None, "auto"):
            name = str(self.manifest.get("dtype") or "").upper()
            return {"BF16": torch.bfloat16, "F16": torch.float16}.get(name, torch.float32)
        if dtype not in table:
            raise ValueError("unknown dtype %r" % dtype)
        return table[dtype]

    def _shard_paths(self):
        files = self.manifest.get("files", {})
        paths = {}
        for group in ("embed", "head", "other"):
            if group in files:
                paths[group] = self.model_dir / files[group]
        for index, name in enumerate(files.get("layer_files", [])):
            paths["layer:%d" % index] = self.model_dir / name
        return paths

    def layer_sizes(self):
        return list(self.manifest.get("layer_bytes", []))

    def _load_state(self, group):
        path = self.paths.get(group)
        if path is None or not path.exists():
            return {}
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")

    def _assign(self, module, state):
        """Copy every tensor in ``state`` onto the matching parameter."""
        placed = 0
        for key, tensor in state.items():
            path, _, attr = key.rpartition(".")
            try:
                target = module.get_submodule(path) if path else module
            except AttributeError:
                continue
            param = getattr(target, attr, None)
            if param is None:
                continue
            param.data = tensor.to(device=self.device, dtype=self.dtype)
            placed += 1
        return placed

    def _release(self, module):
        empty = self.torch.empty(0, device="meta")
        for param in module.parameters():
            param.data = empty

    def _emit(self, kind, index=None, text=""):
        sink = self._sink
        if sink is not None:
            sink.put(Event(kind, index, text))

    def _hooks(self, index):
        def pre(module, inputs):
            state = self._load_state("layer:%d" % index)
            if not state:
                raise RuntimeError("missing shard for layer %d" % index)
            self._assign(module, state)
            self._emit("layer_start", index=index)

        def post(module, inputs, output):
            self._release(module)
            self._emit("layer_done", index=index)

        return pre, post

    def build(self):
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        self.config = AutoConfig.from_pretrained(str(self.model_dir))
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        with init_empty_weights(include_buffers=False):
            model = AutoModelForCausalLM.from_config(self.config, torch_dtype=self.dtype)

        blocks = find_layers(model)
        if not blocks:
            raise RuntimeError(
                "could not find decoder layers; %s is not a supported architecture"
                % self.manifest.get("repo")
            )
        if len(blocks) != self.layer_count:
            raise RuntimeError(
                "shards have %d layers but the config builds %d"
                % (self.layer_count, len(blocks))
            )

        for group in ("embed", "head", "other"):
            state = self._load_state(group)
            if state:
                self._assign(model, state)

        if getattr(self.config, "tie_word_embeddings", False):
            try:
                head = model.get_output_embeddings()
                embed = model.get_input_embeddings()
                if head is not None and embed is not None and head is not embed:
                    head.weight = embed.weight
            except Exception:
                pass

        for index, block in enumerate(blocks):
            pre, post = self._hooks(index)
            block.register_forward_pre_hook(pre)
            block.register_forward_hook(post)

        model.eval()
        try:
            model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        except Exception:
            pass

        self.model = model
        self._blocks = blocks
        return self

    # ------------------------------------------------------------------ run

    def stream(self, messages, max_new_tokens=None, temperature=0.7, top_p=0.9):
        """Yield Event objects: layer_start / layer_done / token / end / error."""
        torch = self.torch
        from transformers import TextIteratorStreamer

        if self.model is None:
            self.build()

        limit = int(max_new_tokens or self.max_new_tokens)
        sink = queue.Queue()
        self._sink = sink

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = "\n".join(m["content"] for m in messages) + "\n"

        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.device)
        attention = encoded.get("attention_mask")
        if attention is not None:
            attention = attention.to(self.device)

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=900.0
        )

        kwargs = {
            "input_ids": input_ids,
            "streamer": streamer,
            "max_new_tokens": limit,
            "do_sample": temperature > 0,
            "temperature": max(float(temperature), 1e-4),
            "top_p": float(top_p),
            "use_cache": True,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if attention is not None:
            kwargs["attention_mask"] = attention

        def work():
            try:
                with torch.inference_mode():
                    self.model.generate(**kwargs)
            except Exception as exc:
                sink.put(Event("error", text="%s: %s" % (type(exc).__name__, exc)))
                try:
                    streamer.end()
                except Exception:
                    pass

        def pump():
            try:
                for chunk in streamer:
                    if chunk:
                        sink.put(Event("token", text=chunk))
            finally:
                sink.put(Event("end"))

        threading.Thread(target=work, daemon=True).start()
        threading.Thread(target=pump, daemon=True).start()

        while True:
            event = sink.get()
            if event.type == "error":
                yield event
                break
            if event.type == "end":
                break
            yield event

    def close(self):
        self._sink = None
        self.model = None
        self._blocks = []
        try:
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()
        except Exception:
            pass
