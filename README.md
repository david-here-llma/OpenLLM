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

## Install: one command

```bash
curl -fsSL https://raw.githubusercontent.com/david-here-llma/OpenLLM/main/get.sh | bash
```

That single command installs everything and then starts OpenLLM:

- on **Termux**: git/curl/proot-distro, then an Ubuntu sandbox, then python,
  torch, transformers and accelerate inside it
- on **Linux**: system packages, then a venv with torch and transformers
- writes the `openllm` command
- launches the app

Afterwards, forever after:

```bash
openllm
```

Re-running the install command is safe - every step is skipped if it is already
done, so it doubles as a repair command. First run takes 5-15 minutes because
PyTorch is a few hundred MB.

## Use

```bash
openllm list                 # catalog, 135M up to 70B
openllm pull qwen2.5-0.5b    # download + split into layer shards
openllm chat qwen2.5-0.5b    # stream a reply, one layer at a time
openllm chat                 # pick from what is installed
openllm rm qwen2.5-0.5b
openllm doctor
```

Chat flags: `--device cpu|cuda`, `--dtype auto|float32|float16|bfloat16`,
`--max-new-tokens 256`, `--temperature 0.7`, `--mock`.

In chat: `/clear` resets history, `/exit` quits.

Start small - 0.5B to 1.5B on a phone. `pull` needs about twice the model size
free (raw download and shards coexist during the split).

Gated repos (Llama, Mistral) need `export HF_TOKEN=...` after accepting the
licence on the model page.

## What is real

| Piece | Status |
| --- | --- |
| `openllm pull` - downloads safetensors from the Hub | real |
| `openllm/splitter.py` - groups tensors by layer, one file per layer + manifest | real |
| `openllm/runner.py` - meta skeleton, per-layer load / run / release via hooks | real |
| `openllm/ui.py` - boxes, timer, percentage, tok/s | real, driven by runner events |
| `openllm/mock.py` - simulated tokens when torch is missing | mock |
| `get.sh` - one-command installer | real |

## Where things go

```
~/.openllm/models/<owner>__<repo>/
    manifest.json          layer count, byte sizes, tensor totals
    layer_00.safetensors   one file per transformer layer
    embed.safetensors  head.safetensors  other.safetensors
    config.json  tokenizer*  ...
```

Set `OPENLLM_HOME` to move that tree.

## Manual install

If you already have torch and transformers, skip `get.sh`:

```bash
git clone https://github.com/david-here-llma/OpenLLM.git ~/OpenLLM
pip install -r ~/OpenLLM/requirements.txt
export PYTHONPATH="$HOME/OpenLLM:$PYTHONPATH"
python3 -m openllm list
```

## Supported models

The catalog covers decoder architectures that expose `model.layers`
(Qwen2/2.5, Llama 3.x, Mistral, SmolLM2, Phi-3.5) or `transformer.h` (GPT-2
family). `openllm pull owner/repo` accepts any other repo, but the split needs a
recognisable `layers.N.` tensor naming scheme.

## Honest status

Written without a test run: the machine it was built on was unreachable, so no
part of this has been executed end to end. The design is deliberate (meta
skeleton plus per-parameter assignment from shards, which avoids the usual
`to_empty` buffer bug), and the installer is idempotent, but expect to debug the
first `openllm pull` on your own hardware - open an issue with the traceback.

MIT licensed.
