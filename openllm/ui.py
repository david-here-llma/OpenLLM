"""Terminal renderer: layer boxes driven by real layer-load events."""

import shutil
import sys
import time
import textwrap

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


def human_bytes(value):
    value = float(value or 0)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return ("%.0f %s" if unit == "B" else "%.2f %s") % (value, unit)
        value /= 1024.0
    return "%.2f GB" % value


def elapsed_str(seconds):
    if seconds < 60:
        return "%5.1fs" % seconds
    return "%dm%02ds" % (int(seconds) // 60, int(seconds) % 60)


class Renderer:
    """Draws the layer grid, the live timer and the completion percentage."""

    def __init__(self, layer_count, model_label, max_tokens, out=None, fps=25):
        self.layer_count = max(1, int(layer_count))
        self.model_label = model_label
        self.max_tokens = max(1, int(max_tokens))
        self.out = out or sys.stdout
        self.interval = 1.0 / max(1, fps)
        self.layer = 0
        self.tokens = 0
        self.text = ""
        self.phase = "starting"
        self.started = None
        self._last = 0.0
        self._alt = False

    # ------------------------------------------------------------------ lifecycle

    def _tty(self):
        try:
            return self.out.isatty()
        except Exception:
            return False

    def __enter__(self):
        self.started = time.monotonic()
        if self._tty():
            self.out.write(ENTER_ALT + HIDE_CURSOR)
            self.out.flush()
            self._alt = True
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._alt:
            self.out.write(EXIT_ALT + SHOW_CURSOR)
            self.out.flush()
            self._alt = False
        return False

    # ------------------------------------------------------------------ events

    def on_layer(self, index):
        self.layer = int(index)
        self.phase = "decode"
        self.draw()

    def on_token(self, text):
        self.tokens += 1
        self.text += text
        self.layer = self.layer_count - 1
        self.draw(force=True)

    def set_phase(self, phase):
        self.phase = phase
        self.draw(force=True)

    def elapsed(self):
        return (time.monotonic() - self.started) if self.started else 0.0

    # ------------------------------------------------------------------ drawing

    @staticmethod
    def size():
        info = shutil.get_terminal_size((80, 24))
        return max(34, info.columns), max(18, info.lines)

    def draw(self, force=False):
        now = time.monotonic()
        if not force and (now - self._last) < self.interval:
            return
        self._last = now

        if not self._tty():
            return

        width, height = self.size()
        elapsed = self.elapsed()
        pct = min(100, int(round(self.tokens * 100.0 / self.max_tokens)))
        spin = SPIN[int(elapsed * 9) % len(SPIN)]
        rule = "-" * min(width - 6, 72)

        lines = [
            "",
            "  " + BOLD + ACCENT + "OpenLLM" + RESET + "  " + GREY
            + self.model_label + " | " + str(self.layer_count) + " layers | real shards"
            + RESET,
            "  " + FAINT + rule + RESET,
            "",
        ]

        area = max(3, height - 12)
        rows = textwrap.wrap(self.text, width=max(24, width - 8)) if self.text else []
        rows = rows[-area:]
        for row in rows:
            lines.append("  " + TEXT + row + RESET)
        for _ in range(area - len(rows)):
            lines.append("")

        lines.append("")
        per_row = max(6, min(self.layer_count, max(6, (width - 8) // 2)))
        for start in range(0, self.layer_count, per_row):
            cells = []
            for i in range(start, min(start + per_row, self.layer_count)):
                if i < self.layer:
                    cells.append(GREEN + CELL_ON + RESET)
                elif i == self.layer:
                    cells.append(BOLD + AMBER + CELL_ON + RESET)
                else:
                    cells.append(FAINT + CELL_OFF + RESET)
            lines.append("  " + "".join(cells))

        bar_w = max(10, width - 42)
        filled = int(round(bar_w * pct / 100.0))
        bar = GREEN + BAR_ON * filled + RESET + FAINT + BAR_OFF * (bar_w - filled) + RESET
        rate = (self.tokens / elapsed) if elapsed > 0.01 else 0.0

        lines.append("")
        lines.append(
            "  " + GREY + "elapsed" + RESET + " " + TEXT + elapsed_str(elapsed) + RESET
            + "  " + bar + " " + BOLD + ACCENT + ("%3d%%" % pct) + RESET
            + "  " + GREY + str(self.tokens) + "/" + str(self.max_tokens) + " tok" + RESET
        )
        lines.append(
            "  " + GREY + self.phase + RESET + " " + AMBER + spin + RESET
            + "  " + GREY + "layer " + str(min(self.layer + 1, self.layer_count))
            + "/" + str(self.layer_count) + RESET
            + "  " + GREY + "%.1f tok/s" % rate + RESET
        )

        self.out.write(CLEAR + "\n".join(lines))
        self.out.flush()

    def flush_text(self):
        """Print the finished reply to the normal screen (keeps scrollback)."""
        width, _ = self.size()
        self.out.write("\n")
        self.out.write("  " + ACCENT + "openllm" + RESET + "  " + GREY
                       + elapsed_str(self.elapsed()) + " | " + str(self.tokens)
                       + " tok" + RESET + "\n")
        for line in textwrap.wrap(self.text, width=max(24, width - 8)) or [""]:
            self.out.write("  " + TEXT + line + RESET + "\n")
        self.out.write("\n")
        self.out.flush()
