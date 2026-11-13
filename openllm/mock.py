"""Simulated backend, used when torch/transformers are unavailable.

It drives the same Renderer as the real engine, so the UI is identical;
only the timings and the tokens are invented.
"""

import random
import re
import time

from .ui import Renderer

OPENERS = [
    "The simulated backend answered because the real runtime is not installed.",
    "This reply is generated locally with no model weights at all.",
]

BODIES = [
    "Install torch, transformers and accelerate to stream a real checkpoint layer by layer.",
    "Each box on the layer row is one transformer layer; in a real run it lights up when that shard is loaded into memory.",
    "Run 'openllm pull <key>' to download a model from Hugging Face and split it into per-layer shards.",
    "Run 'openllm doctor' to see which pieces of the runtime are missing.",
]

CLOSERS = [
    "Use /exit when you are done.",
    "Type another message to watch the sweep again.",
]


def fake_reply(text):
    snippet = " ".join(text.split())
    if len(snippet) > 56:
        snippet = snippet[:53] + "..."
    bodies = random.sample(BODIES, 2)
    return " ".join([
        'You said: "%s".' % snippet,
        random.choice(OPENERS),
        bodies[0],
        bodies[1],
        random.choice(CLOSERS),
    ])


def chat(model_label="mock", max_new_tokens=256, speed=1.0, layer_count=28):
    print("\n  simulated backend - no weights are loaded\n")
    while True:
        try:
            raw = input("  you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw:
            continue
        if raw.lower() in ("/exit", "/quit", "exit", "quit"):
            break

        reply = fake_reply(raw)
        tokens = re.findall(r"\S+\s*", reply) or [reply]
        per_token = 0.15 / max(0.2, speed)

        view = Renderer(layer_count, model_label, max(1, min(len(tokens), max_new_tokens)))
        with view:
            view.set_phase("prefill")
            for _ in range(layer_count):
                view.on_layer(_)
            for token in tokens:
                for index in range(layer_count):
                    view.on_layer(index)
                    time.sleep(per_token / layer_count)
                view.on_token(token)
            view.set_phase("done")
        view.flush_text()
    return 0
