from pathlib import Path

import numpy as np
import pytest

from tests.test_screen_text import CLASSES, _font, _screen

ROM, VIDEO = Path("pokeyellow.gbc"), Path("data/longplay/video.h5")


def test_a_drawn_party_menu_parses():
    from poke.screens import party_screen

    font = _font()
    frame = _screen(font, {(0, 2): "PIKACHU", (0, 13): "16", (1, 13): "40/ 40",
                           (2, 2): "PIDGEY", (2, 13): "16", (2, 17): "PAR", (3, 13): "41/ 41",
                           (14, 1): "Choose a POK"})
    party = party_screen(frame, CLASSES, font)
    assert party == [{"name": "PIKACHU", "level": 16, "hp": 40, "max_hp": 40, "status": None},
                     {"name": "PIDGEY", "level": 16, "hp": 41, "max_hp": 41, "status": "PAR"}]


def test_no_party_menu_without_its_prompt():
    from poke.screens import party_screen

    font = _font()
    frame = _screen(font, {(0, 2): "PIKACHU", (0, 13): "16", (1, 13): "40/ 40"})
    assert party_screen(frame, CLASSES, font) is None


@pytest.mark.skipif(not (ROM.exists() and VIDEO.exists()), reason="needs the ROM and the video")
def test_the_videos_party_and_bag_screens_parse():
    import h5py

    from poke.screen_text import font
    from poke.screens import bag_screen, party_screen
    from poke.symbols import load
    from scripts.sync_opening import text_classes, video_frames

    glyphs = font(ROM.read_bytes(), load())
    with h5py.File(VIDEO) as h5:
        classes = text_classes(h5["colours"][:])
        celadon = video_frames(h5, 659900, 1)[0]
        mtmoon = video_frames(h5, 153300, 1)[0]
        viridian = video_frames(h5, 1315620, 1)[0]
    party = party_screen(celadon, classes, glyphs)
    assert [p["level"] for p in party] == [26, 27, 28, 26, 26, 25]
    assert party[1]["name"] == "PIDGEOTTO" and party[1]["status"] == "PAR"
    assert party[0]["hp"] == 24 and party[0]["max_hp"] == 74
    assert [p["name"] for p in party_screen(mtmoon, classes, glyphs)] == [
        "PIKACHU", "PIDGEY", "NIDORAN♀", "MAGIKARP"]
    assert "BICYCLE" in bag_screen(viridian, classes, glyphs)
