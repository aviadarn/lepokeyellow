"""Game state the screen shows: the party menu and the bag's item list.

Read with `poke.screen_text`, from a video frame or an emulator frame. A
window of the longplay replayed from a door's save state has that state's
party and bag, not the player's; these are what the video shows of them.
"""

from __future__ import annotations

import re

import numpy as np

from poke.screen_text import read_row

PROMPTS = ("Choose a POK", "Move POK", "Use item on", "which POK")
STATUSES = ("PAR", "SLP", "PSN", "BRN", "FRZ")
NAME = re.compile(r"^[A-Z][A-Z♀♂.'\- ]*$")


def party_screen(frame: np.ndarray, classes: np.ndarray, glyphs: np.ndarray) -> list[dict] | None:
    """[{name, level, hp, max_hp, status}] per member shown, top to bottom,
    or None when the frame is not the party menu (its prompt is missing or
    no member reads). Each member is two tile rows: the name in columns
    2-11, ":L" and the level in 12-15, a status in 17-19; then the HP."""
    prompt = " ".join(read_row(frame, classes, glyphs, r) for r in range(13, 17))
    if not any(p in prompt for p in PROMPTS):
        return None
    party = []
    for slot in range(6):
        top = read_row(frame, classes, glyphs, 2 * slot)
        bottom = read_row(frame, classes, glyphs, 2 * slot + 1)
        name = top[2:12].strip(" ?▶▷")
        level = re.search(r"(\d+)", top[13:16])
        hp = re.search(r"(\d+)/\s*(\d+)", bottom[12:20])
        if not (name and NAME.match(name) and level and hp):
            break
        status = top[17:20].strip()
        party.append({"name": name, "level": int(level.group(1)), "hp": int(hp.group(1)),
                      "max_hp": int(hp.group(2)), "status": status if status in STATUSES else None})
    return party or None


def bag_screen(frame: np.ndarray, classes: np.ndarray, glyphs: np.ndarray) -> list[str] | None:
    """Item names listed in the bag's menu, top to bottom down to CANCEL,
    or None when no list item reads. The list sits in a box on the right:
    a cursor column, then the name."""
    names = []
    for row in range(1, 13):
        text = read_row(frame, classes, glyphs, row)
        match = re.search(r"[▶▷ ]([A-Z][A-Z0-9é.'\- ]{2,})", text[4:19])
        if not match:
            continue
        name = match.group(1).strip()
        if name == "CANCEL":
            break
        if NAME.match(name) or name.replace("é", "E").isalnum():
            names.append(name)
    return names or None
