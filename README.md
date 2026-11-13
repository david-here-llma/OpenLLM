# OpenLLM

Layer-streamed inference for Hugging Face models. Instead of holding a whole
checkpoint in memory, OpenLLM splits it into one shard per transformer layer and
loads exactly one layer at a time while it runs.

```
  OpenLLM  Qwen2.5 3B | 36 layers | real shards
  ----------------------------------------------

  The capital of France is Paris, which has been
  the country's centre since...

  ▰▰▰▰▰▰▰▰▰▰▰▰▰▱▱▱▱▱▱▱▱▱▱▱▱▱▱▱▱▱▱▱▱▱▱▱

  elapsed   8.2s  ████░░░░░░░░░░  22%  56/256 tok
  decode ⠹  layer 13/36  6.8 tok/s
```

Each box is a real transformer layer. It turns green when that shard has been
loaded, run, and released - the row is driven by forward hooks, not a timer.

## What is real

| Piece | Status |
| --- | --- |
| `openllm pull` - downloads safetensors from the Hub | real |
| `openllm/splitter.py` - groups tensors by layer, writes one file per layer + manifest | real |
| `openllm/runner.py` - meta skeleton, per-layer load / run / release via hooks | real |
| `openllm/ui.py` - boxes, timer, percentage, tok/s | real, driven by runner events |
| `openllm/mock.py` - simulated tokens when torch is missing | mock |

## Install

```bash
pkg update -y && pkg install -y python git curl
curl -fsSL https://raw.githubusercontent.com/david-here-llma/OpenLLM/main/install.sh | bash
openllm doctor
```

Or by hand:

```bash
git clone https://github.com/david-here-llma/OpenLLM.git ~/.openllm
export PYTHONPATH="$HOME/.openllm:$PYTHONPATH"
python3 -m openllm list
```

### Termux caveat

PyTorch has no official wheel for Android/aarch64, so `pip install torch` fails in
plain Termux. The CLI itself works (`list`, `doctor`, `chat --mock`), but real
inference needs one of:

- `pkg install proot-distro && proot-distro install ubuntu`, then run OpenLLM inside Ubuntu
- a Linux desktop with the same commands

The layer splitting itself only needs `safetensors` + `torch`.

## Commands

```bash
openllm list                 # catalog, 135M up to 70B
openllm pull qwen2.5-3b      # download + split into layer shards
openllm chat qwen2.5-3b      # stream a reply, one layer at a time
openllm chat                 # pick from what is installed
openllm rm qwen2.5-3b
openllm doctor
```

Chat flags: `--device cpu|cuda`, `--dtype auto|float32|float16|bfloat16`,
`--max-new-tokens 256`, `--temperature 0.7`, `--mock`.

In chat: `/clear` resets history, `/exit` quits.

## Where things go

```
~/.openllm/models/<owner>__<repo>/
    manifest.json          layer count, byte sizes, tensor totals
    layer_00.safetensors   one file per transformer layer
    embed.safetensors  head.safetensors  other.safetensors
    config.json  tokenizer*  ...
```

Set `OPENLLM_HOME` to move that tree. Set `HF_TOKEN` for gated repos
(Llama, Mistral).

## Supported models

The catalog covers standard decoder architectures that expose `model.layers`
(Qwen2/2.5, Llama 3.x, Mistral, SmolLM2, Phi-3.5) or `transformer.h` (GPT-2
family). `openllm pull owner/repo` also accepts any other repo, but the split
needs a recognisable `layers.N.` tensor naming scheme.

## Honest status

This is a first real cut, written without a test run: the machine it was built on
was unreachable, so nothing here has been executed end to end. The design is
sound (meta skeleton plus per-parameter assignment from shards avoids the usual
`to_empty` buffer bug), but expect to debug the first `openllm pull` on your own
hardware - open an issue with the traceback.

MIT licensed.
