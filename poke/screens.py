"""Game state the screen shows: the party menu, its submenu, and the bag's item list.

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
START_MENU = ("POKéDEX", "POKéMON", "ITEM", "SAVE", "OPTION", "EXIT")  # a list with these is the START menu
NAME = re.compile(r"^[A-Z][A-Z♀♂.'\- ]*$")
SUBMENU = ("STATS", "SWITCH", "CANCEL")
FIELD_MOVES = ("CUT", "FLY", "SURF", "STRENGTH", "FLASH", "DIG", "TELEPORT", "SOFTBOILED")


def party_screen(frame: np.ndarray, classes: np.ndarray, glyphs: np.ndarray) -> list[dict] | None:
    """[{name, level, hp, max_hp, status}] per member shown, top to bottom,
    or None when the frame is not the party menu (its prompt is missing or
    no member reads). Each member is two tile rows: its icon in columns 1-2,
    the name in 3-12, ":L" and the level from 13, a status in 17-19; then
    the HP."""
    prompt = " ".join(read_row(frame, classes, glyphs, r) for r in range(13, 17))
    if not any(p in prompt for p in PROMPTS):
        return None
    party = []
    for slot in range(6):
        top = read_row(frame, classes, glyphs, 2 * slot)
        bottom = read_row(frame, classes, glyphs, 2 * slot + 1)
        name = top[3:13].strip(" ?▶▷")
        level = re.search(r"(\d+)", top[13:17])
        hp = re.search(r"(\d+)/\s*(\d+)", bottom[12:20])
        if not (name and NAME.match(name) and level and hp):
            break
        status = top[17:20].strip()
        party.append({"name": name, "level": int(level.group(1)), "hp": int(hp.group(1)),
                      "max_hp": int(hp.group(2)), "status": status if status in STATUSES else None})
    return party or None


def party_submenu(frame: np.ndarray, classes: np.ndarray, glyphs: np.ndarray) -> tuple[int, list[str]] | None:
    """(slot, field moves top to bottom) when the party menu's submenu is
    open, else None. The submenu lists the selected member's field moves,
    then STATS, SWITCH and CANCEL, on every other row; the party's cursor
    marks the member in column 0 of its HP row (2 * slot + 1)."""
    rows = [read_row(frame, classes, glyphs, r) for r in range(18)]
    last = [(re.findall(r"[A-Z]+", row[9:]) or [""])[-1] for row in rows]
    stats = next((r for r in range(14) if last[r:r + 5:2] == list(SUBMENU)), None)
    cursor = next((r for r in range(1, 12, 2) if rows[r][0] in "▶▷"), None)
    if stats is None or cursor is None:
        return None
    moves = []
    for r in range(stats - 2, -1, -2):
        if last[r] not in FIELD_MOVES:
            break
        moves.insert(0, last[r])
    return (cursor - 1) // 2, moves


def bag_screen(frame: np.ndarray, classes: np.ndarray, glyphs: np.ndarray) -> list[tuple[str, int]] | None:
    """(name, quantity) per item listed in the bag's menu, top to bottom down
    to CANCEL, or None when no list item reads, or the list is the START
    menu (whose POKéDEX is an item's name too) or a shop's (a price under
    each name). The list sits in a box on the right: a cursor column, then
    the name; a quantity "×N" on the row under it, none for a key item."""
    items = []
    rows = [read_row(frame, classes, glyphs, row) for row in range(0, 14)]
    for row in range(1, 13):
        match = re.search(r"[▶▷ ]([A-Z][A-Z0-9é.'\- ]{2,})", rows[row][4:19])
        if not match:
            continue
        name = match.group(1).strip()
        if name in START_MENU or "¥" in rows[row + 1]:
            return None
        if name == "CANCEL":
            break
        if NAME.match(name) or name.replace("é", "E").isalnum():
            quantity = re.search(r"×\s*(\d+)", rows[row + 1])
            items.append((name, int(quantity.group(1)) if quantity else 1))
    return items or None
