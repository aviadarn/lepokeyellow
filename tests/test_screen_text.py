from pathlib import Path

import numpy as np
import pytest

from poke.screen_text import CHARS, read_row

PAPER, INK = 3, 0
CLASSES = np.full(256, 2, np.int8)
CLASSES[PAPER], CLASSES[INK] = 0, 1


def _font(seed=0):
    """Distinct random 8x8 glyphs for codes 0x80-0xFF (index = code - 0x80)."""
    rng = np.random.default_rng(seed)
    return rng.random((128, 8, 8)) < 0.35


def _screen(font, text_at):
    frame = np.full((144, 160), PAPER, np.uint8)
    codes = {c: k for k, c in CHARS.items()}
    for (row, col), text in text_at.items():
        for i, ch in enumerate(text):
            if ch == " ":
                continue
            glyph = font[codes[ch] - 0x80]
            tile = frame[row * 8:row * 8 + 8, (col + i) * 8:(col + i) * 8 + 8]
            tile[glyph] = INK
    return frame


def test_glyphs_drawn_on_paper_read_back():
    font = _font()
    frame = _screen(font, {(3, 2): "RED", (5, 0): "L26 24/74"})
    assert read_row(frame, CLASSES, font, 3).strip() == "RED"
    assert read_row(frame, CLASSES, font, 5).strip() == "L26 24/74"
    assert read_row(frame, CLASSES, font, 7).strip() == ""


def test_a_tile_off_by_more_than_the_tolerance_is_unknown():
    font = _font()
    frame = _screen(font, {(3, 2): "R"})
    tile = frame[24:32, 16:24]
    flip = np.argwhere(np.ones((8, 8), bool))[:5]
    for y, x in flip:
        tile[y, x] = INK if tile[y, x] == PAPER else PAPER
    assert read_row(frame, CLASSES, font, 3, tolerance=4)[2] == "?"
    assert read_row(frame, CLASSES, font, 3, tolerance=5)[2] == "R"


ROM, VIDEO = Path("pokeyellow.gbc"), Path("data/longplay/video.h5")


@pytest.mark.skipif(not (ROM.exists() and VIDEO.exists()), reason="needs the ROM and the video")
def test_the_videos_party_screen_reads():
    import h5py

    from poke.screen_text import font
    from poke.symbols import load
    from scripts.sync_opening import text_classes, video_frames

    with h5py.File(VIDEO) as h5:
        frame = video_frames(h5, 659900, 1)[0]
        classes = text_classes(h5["colours"][:])
    glyphs = font(ROM.read_bytes(), load())
    assert "WARTORTLE" in read_row(frame, classes, glyphs, 0)
    assert "PIDGEOTTO" in read_row(frame, classes, glyphs, 2)
