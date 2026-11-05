from pathlib import Path

import numpy as np
import pytest

from tests.test_screen_text import CLASSES, _font, _screen

ROM, VIDEO = Path("pokeyellow.gbc"), Path("data/longplay/video.h5")


def test_a_drawn_party_menu_parses():
    from poke.screens import party_screen

    font = _font()
    frame = _screen(font, {(0, 3): "CHARMELEON", (0, 14): "16", (1, 13): "40/ 40",
                           (2, 3): "PIDGEY", (2, 14): "16", (2, 17): "PAR", (3, 13): "41/ 41",
                           (14, 1): "Choose a POK"})
    party = party_screen(frame, CLASSES, font)
    assert party == [{"name": "CHARMELEON", "level": 16, "hp": 40, "max_hp": 40, "status": None},
                     {"name": "PIDGEY", "level": 16, "hp": 41, "max_hp": 41, "status": "PAR"}]


def test_no_party_menu_without_its_prompt():
    from poke.screens import party_screen

    font = _font()
    frame = _screen(font, {(0, 3): "PIKACHU", (0, 14): "16", (1, 13): "40/ 40"})
    assert party_screen(frame, CLASSES, font) is None


def test_a_bag_list_over_the_start_menu_parses_with_its_quantities():
    # The bag opens over the START menu, whose EXIT stays in sight below it.
    from poke.screens import bag_screen

    font = _font()
    frame = _screen(font, {(4, 6): "BICYCLE", (6, 6): "POTION", (7, 14): "× 3",
                           (8, 6): "CANCEL", (14, 12): "EXIT"})
    assert bag_screen(frame, CLASSES, font) == [("BICYCLE", 1), ("POTION", 3)]


def test_the_start_menu_is_not_a_bag_list():
    from poke.screens import bag_screen

    font = _font()
    frame = _screen(font, {(2, 12): "POKéDEX", (4, 12): "POKéMON", (6, 12): "ITEM",
                           (8, 12): "RED", (10, 12): "SAVE", (12, 12): "OPTION"})
    assert bag_screen(frame, CLASSES, font) is None


def test_a_shop_list_is_not_the_bag():
    from poke.screens import bag_screen

    font = _font()
    frame = _screen(font, {(4, 6): "POTION", (5, 14): "¥300", (6, 6): "CANCEL"})
    assert bag_screen(frame, CLASSES, font) is None


def test_a_drawn_party_submenu_names_its_mon_and_field_moves():
    from poke.screens import party_submenu

    font = _font()
    frame = _screen(font, {(10, 3): "IVYSAUR", (11, 0): "▷", (10, 12): "▶CUT",
                           (12, 13): "STATS", (14, 13): "SWITCH", (16, 13): "CANCEL"})
    assert party_submenu(frame, CLASSES, font) == (5, ["CUT"])
    no_moves = _screen(font, {(1, 0): "▷", (12, 13): "STATS", (14, 13): "SWITCH",
                              (16, 13): "CANCEL"})
    assert party_submenu(no_moves, CLASSES, font) == (0, [])
    assert party_submenu(_screen(font, {(10, 3): "IVYSAUR"}), CLASSES, font) is None


@pytest.mark.skipif(not (ROM.exists() and VIDEO.exists()), reason="needs the ROM and the video")
def test_the_videos_party_and_bag_screens_parse():
    import h5py

    from poke.screen_text import font
    from poke.screens import bag_screen, party_screen, party_submenu
    from poke.symbols import load
    from scripts.sync_opening import text_classes, video_frames

    glyphs = font(ROM.read_bytes(), load())
    with h5py.File(VIDEO) as h5:
        classes = text_classes(h5["colours"][:])
        celadon = video_frames(h5, 659900, 1)[0]
        mtmoon = video_frames(h5, 153300, 1)[0]
        viridian = video_frames(h5, 1315620, 1)[0]
        bag_over_start = video_frames(h5, 1315600, 1)[0]
        submenu = video_frames(h5, 659922, 1)[0]
    party = party_screen(celadon, classes, glyphs)
    assert [p["level"] for p in party] == [26, 27, 28, 26, 26, 25]
    assert party[1]["name"] == "PIDGEOTTO" and party[1]["status"] == "PAR"
    assert party[0]["hp"] == 24 and party[0]["max_hp"] == 74
    assert party[2]["name"] == "CHARMELEON"            # ten letters: columns 3-12
    assert [p["name"] for p in party_screen(mtmoon, classes, glyphs)] == [
        "PIKACHU", "PIDGEY", "NIDORAN♀", "MAGIKARP"]
    assert ("BICYCLE", 1) in bag_screen(viridian, classes, glyphs)
    assert bag_screen(bag_over_start, classes, glyphs) == [
        ("BICYCLE", 1), ("SUPER POTION", 1), ("GREAT BALL", 6), ("REVIVE", 11)]
    assert party_submenu(submenu, classes, glyphs) == (5, ["CUT"])
