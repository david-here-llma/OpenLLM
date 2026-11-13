#!/usr/bin/env python3
"""OpenLLM - a mock terminal runtime for layer-streamed inference.

This build is the UI skeleton only: it draws the layer boxes, the live
elapsed timer and the completion percentage, but every token and every
timing in here is simulated locally. No model weights are loaded and
nothing is sent over the network.
"""

import argparse
import random
import re
import shutil
import sys
import textwrap
import time

VERSION = "0.1.0"

RESET = "\x1b[0m"
BOLD = "\x1b[1m"
ENTER_ALT = "\x1b[?1049h"
EXIT_ALT = "\x1b[?1049l"
HIDE_CURSOR = "\x1b[?25l"
SHOW_CURSOR = "\x1b[?25h"
CLEAR = "\x1b[H\x1b[2J"

ACCENT = "\x1b[38;5;45m"
GREEN = "\x1b[38;5;42m"
AMBER = "\x1b[38;5;214m"
TEXT = "\x1b[38;5;252m"
GREY = "\x1b[38;5;244m"
FAINT = "\x1b[38;5;238m"

CELL_ON = "\u25b0"
CELL_OFF = "\u25b1"
BAR_ON = "\u2588"
BAR_OFF = "\u2591"
SPIN = "\u280b\u2819\u2839\u2838\u283c\u2834\u2826\u2827\u2807\u280f"

OPENERS = [
    "The mock backend picked the message up and started streaming straight away.",
    "This run is a local simulation, so there is no GPU, no weights and no network involved.",
    "Request received and queued on the fake runtime.",
]

BODIES = [
    "Each box on the layer row is one transformer layer, and it lights up as that layer finishes its slice of work for the token being decoded.",
    "The bar underneath tracks how much of the reply has been emitted, so the percentage and the token counter move together.",
    "Real streaming spends most of its time pulling weights off storage, which is why the layers advance steadily while the percentage crawls on a phone.",
    "On an 8 GB device a 3B model at 4-bit quantization is the realistic ceiling, and the timings here are shaped to look like that.",
    "Nothing is cached between runs yet, so every pass starts from a cold layer sweep.",
]

CLOSERS = [
    "Send another message to watch the sweep again.",
    "Use /stats to print the timings of the last run.",
    "Use /exit when you are done.",
]


def elapsed_str(seconds):
    if seconds < 60:
        return "%5.1fs" % seconds
    return "%dm%02ds" % (int(seconds) // 60, int(seconds) % 60)


def make_reply(user_text):
    snippet = " ".join(user_text.split())
    if len(snippet) > 56:
        snippet = snippet[:53] + "..."
    bodies = random.sample(BODIES, 2)
    parts = [
        'You said: "%s".' % snippet,
        random.choice(OPENERS),
        bodies[0],
        bodies[1],
        random.choice(CLOSERS),
    ]
    return " ".join(parts)


class App:
    def __init__(self, layers=28, model="qwen3-mock-7b", speed=1.0):
        self.layers = max(2, layers)
        self.model = model
        self.speed = max(0.2, speed)
        self.last = None

    @staticmethod
    def size():
        s = shutil.get_terminal_size((80, 24))
        return max(34, s.columns), max(18, s.lines)

    def width(self):
        return self.size()[0]

    def banner(self):
        rule = "-" * min(self.width() - 6, 72)
        print()
        print("  " + BOLD + ACCENT + "OpenLLM" + RESET + "  " + GREY + "v" + VERSION + RESET)
        print("  " + FAINT + rule + RESET)
        print("  " + GREY + "model " + RESET + TEXT + self.model + RESET
              + "   " + GREY + "layers " + RESET + TEXT + str(self.layers) + RESET)
        print("  " + GREY + "UI preview only - every token and timing is simulated." + RESET)
        print("  " + GREY + "Type a message. /help for commands, /exit to quit." + RESET)
        print()

    def help(self):
        rows = [
            ("/help", "show this list"),
            ("/stats", "timings of the last run"),
            ("/clear", "clear the screen"),
            ("/exit", "quit"),
        ]
        print()
        for cmd, what in rows:
            print("  " + ACCENT + cmd.ljust(8) + RESET + GREY + what + RESET)
        print("  " + GREY + "flags: --layers N  --model NAME  --speed 2.0" + RESET)
        print()

    def stats(self):
        if not self.last:
            print("  " + GREY + "no runs yet" + RESET)
            return
        secs, ntok = self.last
        rate = (ntok / secs) if secs > 0 else 0.0
        print("  " + GREY + "last run: " + RESET + TEXT + ("%.1fs" % secs)
              + " for " + str(ntok) + " tokens (" + ("%.1f" % rate) + " tok/s)" + RESET)

    def draw(self, prompt, out, layer, elapsed, done, total, phase):
        w, h = self.size()
        pct = int(done * 100 / total) if total > 0 else 0
        spin = SPIN[int(elapsed * 9) % len(SPIN)]
        rule = "-" * min(w - 6, 72)

        one = " ".join(prompt.split())
        lim = w - 14
        if len(one) > lim:
            one = one[: max(0, lim - 3)] + "..."

        lines = [
            "",
            "  " + BOLD + ACCENT + "OpenLLM" + RESET + "  " + GREY
            + "mock runtime | " + self.model + " | " + str(self.layers) + " layers" + RESET,
            "  " + FAINT + rule + RESET,
            "",
            "  " + GREY + "you" + RESET + "    " + TEXT + one + RESET,
            "",
        ]

        area = max(3, h - 13)
        rows = textwrap.wrap(out, width=max(24, w - 8)) if out else []
        rows = rows[-area:]
        for r in rows:
            lines.append("  " + TEXT + r + RESET)
        for _ in range(area - len(rows)):
            lines.append("")

        lines.append("")
        per_row = max(6, min(self.layers, max(6, (w - 8) // 2)))
        for start in range(0, self.layers, per_row):
            cells = []
            for i in range(start, min(start + per_row, self.layers)):
                if i < layer:
                    cells.append(GREEN + CELL_ON + RESET)
                elif i == layer:
                    cells.append(BOLD + AMBER + CELL_ON + RESET)
                else:
                    cells.append(FAINT + CELL_OFF + RESET)
            lines.append("  " + "".join(cells))

        bar_w = max(10, w - 40)
        filled = int(round(bar_w * pct / 100.0))
        bar = GREEN + BAR_ON * filled + RESET + FAINT + BAR_OFF * (bar_w - filled) + RESET
        lines.append("")
        lines.append(
            "  " + GREY + "elapsed" + RESET + " " + TEXT + elapsed_str(elapsed) + RESET
            + "  " + bar + " " + BOLD + ACCENT + ("%3d%%" % pct) + RESET
            + "  " + GREY + str(done) + "/" + str(total) + " tok" + RESET
        )
        lines.append(
            "  " + GREY + phase + RESET + " " + AMBER + spin + RESET
            + "  " + GREY + "layer " + str(min(layer + 1, self.layers))
            + "/" + str(self.layers) + RESET
        )

        sys.stdout.write(CLEAR + "\n".join(lines))
        sys.stdout.flush()

    def _animate(self, prompt):
        reply = make_reply(prompt)
        tokens = re.findall(r"\S+\s*", reply) or [reply]
        total = len(tokens)
        per_token = 0.15 / self.speed
        prefill = 0.7 / self.speed
        began = time.monotonic()
        out = ""
        done = 0

        while True:
            e = time.monotonic() - began
            if e >= prefill:
                break
            f = e / prefill
            self.draw(prompt, "", min(self.layers - 1, int(f * self.layers)),
                      e, 0, total, "prefill")
            time.sleep(0.03)

        for tok in tokens:
            t0 = time.monotonic()
            while True:
                e = time.monotonic() - t0
                if e >= per_token:
                    break
                frac = e / per_token
                self.draw(prompt, out, min(self.layers - 1, int(frac * self.layers)),
                          time.monotonic() - began, done, total, "decode")
                time.sleep(0.02)
            out += tok
            done += 1

        elapsed = time.monotonic() - began
        self.draw(prompt, out, self.layers - 1, elapsed, total, total, "done")
        time.sleep(0.18)
        return out.strip(), elapsed, total

    def stream(self, prompt):
        if not sys.stdout.isatty():
            reply = make_reply(prompt).strip()
            ntok = len(re.findall(r"\S+\s*", reply)) or 1
            print(reply)
            self.last = (0.0, ntok)
            return reply, 0.0, ntok

        sys.stdout.write(ENTER_ALT + HIDE_CURSOR)
        sys.stdout.flush()
        try:
            reply, secs, ntok = self._animate(prompt)
        finally:
            sys.stdout.write(EXIT_ALT + SHOW_CURSOR)
            sys.stdout.flush()
        self.last = (secs, ntok)
        return reply, secs, ntok

    def run(self):
        self.banner()
        while True:
            try:
                raw = input("  " + GREY + "you" + RESET + " " + ACCENT + ">" + RESET + " ")
            except (EOFError, KeyboardInterrupt):
                sys.stdout.write("\n")
                break

            msg = raw.strip()
            if not msg:
                continue
            low = msg.lower()
            if low in ("/exit", "/quit", "exit", "quit", ":q"):
                break
            if low == "/help":
                self.help()
                continue
            if low == "/clear":
                sys.stdout.write(CLEAR)
                self.banner()
                continue
            if low == "/stats":
                self.stats()
                continue

            reply, secs, ntok = self.stream(msg)
            rate = (ntok / secs) if secs > 0 else 0.0
            print("  " + ACCENT + "openllm" + RESET + "  " + GREY
                  + ("%.1fs" % secs) + " | " + str(ntok) + " tok | "
                  + ("%.1f" % rate) + " tok/s" + RESET)
            for line in textwrap.wrap(reply, width=max(24, self.width() - 8)):
                print("  " + TEXT + line + RESET)
            print()

        print("  " + GREY + "bye" + RESET)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="openllm",
        description="OpenLLM mock terminal runtime (UI preview only).",
    )
    parser.add_argument("--model", default="qwen3-mock-7b", help="label shown in the header")
    parser.add_argument("--layers", type=int, default=28, help="number of layer boxes")
    parser.add_argument("--speed", type=float, default=1.0, help="animation speed multiplier")
    parser.add_argument("--version", action="version", version="openllm " + VERSION + " (mock)")
    args = parser.parse_args(argv)

    try:
        App(layers=args.layers, model=args.model, speed=args.speed).run()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
