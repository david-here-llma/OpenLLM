# OpenLLM

A mock terminal runtime that previews how **OpenLLM** will look on a phone: layer-by-layer
streaming, a live elapsed timer, and a completion percentage.

> **This is a UI mockup.** No model weights are downloaded or loaded and no inference runs.
> Every token and every timing you see is generated locally by a simulation.

Built to be readable on a small Termux screen and to run on a 8 GB Android device.

## What you see

```
  OpenLLM  mock runtime | qwen3-mock-7b | 28 layers
  ----------------------------------------------

  you    what can you do

  <reply text streams in here>

  ▰▰▰▰▰▰▰▰▰▱▱▱▱▱▱▱▱▱▱▱▱▱▱▱▱▱▱▱

  elapsed  12.4s  ██████░░░░░░░░░░░░  34%  18/52 tok
  decode ⠹  layer 9/28
```

- **Layer boxes** - one box per transformer layer. Each pass fills them left to right,
  bright yellow for the layer being computed and green for the ones already done.
- **elapsed** - live wall-clock timer for the current message, in seconds then `m:ss`.
- **percent** - how much of the reply has been emitted, next to the token counter.
- **phase** - `prefill` while the prompt is being read, `decode` while tokens stream,
  `done` when the pass finishes.

The animation runs in the terminal's alternate screen, so your scrollback stays clean:
when a reply finishes it is printed normally and stays there.

## Install (Termux)

```bash
pkg update -y && pkg install -y python git curl
curl -fsSL https://raw.githubusercontent.com/david-here-llma/OpenLLM/main/install.sh | bash
openllm
```

Or clone it yourself:

```bash
git clone https://github.com/david-here-llma/OpenLLM.git ~/.openllm
python3 ~/.openllm/openllm.py
```

There are no third-party Python dependencies - standard library only.

## Commands

| Command | What it does |
| --- | --- |
| `/help` | list commands |
| `/stats` | timings of the last run |
| `/clear` | clear the screen |
| `/exit` | quit |

## Flags

```bash
openllm --layers 40 --model my-model --speed 2
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--layers` | `28` | number of layer boxes |
| `--model` | `qwen3-mock-7b` | label in the header |
| `--speed` | `1.0` | animation speed multiplier |

## Notes

- Needs a real TTY for the animation; when output is piped it prints plain text instead.
- Tested on a phone-sized terminal (about 40x30); it also lays out at 80 columns.
- Android may freeze background Termux sessions - run `termux-wake-lock` if a long run stalls.

## Roadmap

1. Swap the simulated token stream for a real `llama.cpp` / AirLLM style backend.
2. Report real per-layer timings instead of shaped ones.
3. A real tokenizer so the token counter matches the model.

MIT licensed.
