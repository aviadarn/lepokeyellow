"""Gen-1 text read straight off a screen.

The game draws text on the screen's 8x8 tile grid with a 1bpp font
(`FontGraphics`, character codes 0x80-0xFF). A tile of a frame whose pixels
are all paper and ink (`scripts.sync_opening.text_classes`) is matched to
the font's glyphs: the closest within `tolerance` differing pixels names the
character. Works on video frames and emulator frames alike.
"""

from __future__ import annotations

import numpy as np

from poke.warp import _offset

# pokered's charmap for the font's codes; the rest of 0x80-0xFF is unused.
CHARS: dict[int, str] = {0x80 + i: chr(ord("A") + i) for i in range(26)}
CHARS.update({0xA0 + i: chr(ord("a") + i) for i in range(26)})
CHARS.update({0xF6 + i: str(i) for i in range(10)})
CHARS.update({0x7F: " ", 0x9A: "(", 0x9B: ")", 0x9C: ":", 0x9D: ";", 0x9E: "[", 0x9F: "]",
              0xBA: "é", 0xE0: "'", 0xE3: "-", 0xE6: "?", 0xE7: "!", 0xE8: ".",
              0xEC: "▷", 0xED: "▶", 0xEE: "▼", 0xEF: "♂", 0xF0: "¥", 0xF1: "×",
              0xF2: ".", 0xF3: "/", 0xF4: ",", 0xF5: "♀"})
FONT_FIRST = 0x80


def font(rom: bytes, symbols: dict[str, tuple[int, int]]) -> np.ndarray:
    """(128, 8, 8) bool, ink, for character codes 0x80-0xFF."""
    bank, address = symbols["FontGraphics"]
    start = _offset(bank, address)
    raw = np.frombuffer(rom[start:start + 128 * 8], np.uint8).reshape(128, 8)
    return np.unpackbits(raw, axis=1).reshape(128, 8, 8).astype(bool)


def read_row(frame: np.ndarray, classes: np.ndarray, glyphs: np.ndarray, row: int,
             col0: int = 0, col1: int = 20, tolerance: int = 4) -> str:
    """Tile row `row` (0-17), columns [col0, col1): a character per tile,
    a space for plain paper, `?` for anything the font does not match."""
    out = []
    codes = [c for c in CHARS if c - FONT_FIRST < len(glyphs)]
    stack = glyphs[[c - FONT_FIRST for c in codes]]
    for col in range(col0, col1):
        tile = classes[frame[row * 8:row * 8 + 8, col * 8:col * 8 + 8]]
        if (tile == 2).any():
            out.append("?")
            continue
        ink = tile == 1
        if not ink.any():
            out.append(" ")
            continue
        wrong = (stack != ink).sum(axis=(1, 2))
        best = int(np.argmin(wrong))
        out.append(CHARS[codes[best]] if wrong[best] <= tolerance else "?")
    return "".join(out)
