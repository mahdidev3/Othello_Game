from __future__ import annotations

import os
import sys
from dataclasses import dataclass


def _supports_color() -> bool:
    if os.getenv("NO_COLOR"):
        return False
    return sys.stdout.isatty()


@dataclass
class Colorizer:
    """Lightweight ANSI color helper with graceful degradation."""

    enabled: bool = _supports_color()

    COLORS = {
        "black": "30",
        "red": "31",
        "green": "32",
        "yellow": "33",
        "blue": "34",
        "magenta": "35",
        "cyan": "36",
        "white": "37",
        "bright_black": "90",
        "bright_white": "97",
        "gray": "90",
    }

    BACKGROUND = {
        "black": "40",
        "white": "47",
        "bright_black": "100",
        "bright_white": "107",
        "green": "42",
    }

    def colorize(self, text: str, fg: str | None = None, bg: str | None = None) -> str:
        if not self.enabled:
            return text
        codes = []
        if fg and fg in self.COLORS:
            codes.append(self.COLORS[fg])
        if bg and bg in self.BACKGROUND:
            codes.append(self.BACKGROUND[bg])
        if not codes:
            return text
        return f"\033[{';'.join(codes)}m{text}\033[0m"
