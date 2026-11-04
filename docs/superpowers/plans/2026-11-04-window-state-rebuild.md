# Rebuild a window's party and bag from the screen

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Sampled windows from door states stop within seconds on state the
state does not have:
- the party (Mt. Moon, Celadon);
- the remembered START-menu cursor, and the bag (Viridian).

Read the party and bag off the video's own screens, write them into the
emulator before the window replays, and resync the menu cursor. Fix the
budget so backtracking cannot reset it at the same wall. Re-run the three
windows.

**Architecture:**
- **Reading text.** `poke/screen_text.py` reads Gen-1 text off a frame:
  - each 8×8 tile of the screen grid is classed paper/ink (the judge's
    classes);
  - it is matched to the ROM's 1bpp font (`FontGraphics`, codes
    0x80–0xFF).
- **Parsing screens.** `poke/screens.py` parses the party menu and the bag
  list into plain records.
- **Writing state.** `poke/party.py` writes them into WRAM, with names and
  ids from the ROM's `MonsterNames`, `ItemNames`, `BaseStats` and
  `PokedexOrder`.
- **Applying it.** `scripts/repair_labels.py --rebuild` scans the window's
  video for the first readable party and bag screens and applies them to the
  anchor state. Everything set is recorded as shown or guessed.
- **Menu cursor.** `MENU_CURSOR` = ((0xCC2D, range(7)),) joins the cheap
  resync tiers.

**Spec:** `docs/superpowers/specs/2026-11-03-generic-video-sync-design.md`
(Part 3: patch what the screen shows, flag the rest as guessed).

## Global Constraints

- **The judge is unchanged.**
- **Everything shown is set as shown.** Hidden values are guessed (DVs 8,
  stat experience 0, starting moves from BaseStats) and listed in the
  report.
- **Windows are debug runs** (`--dry-run`). Nothing is written to
  `labels.h5`.
- **Mac safety** as before. Tests that need the ROM skip without it.

### Task 1: Text off the screen — `poke/screen_text.py`, `tests/test_screen_text.py`

- **Interface:**
  - `font(rom: bytes, symbols) -> np.ndarray`, (128, 8, 8) bool, ink;
  - `CHARS: dict[int, str]`, the Gen-1 charmap for 0x80–0xFF;
  - `read_row(frame, classes, glyphs, row, col0=0, col1=20, tolerance=4) -> str`.
    It gives `?` for a non-paper tile no glyph matches, and a space for paper.
- **Tests:**
  - glyphs drawn into a synthetic paper frame read back;
  - a tile with `tolerance` + 1 flipped pixels reads `?`;
  - on the video (skips without data): window 2's party screen at
    3:03:30 reads "WARTORTLE" and "PIDGEOTTO".

### Task 2: Parse party and bag screens — `poke/screens.py`, `tests/test_screens.py`

- **Interface:**
  - `party_screen(frame, classes, glyphs) -> list[dict] | None`. Each record
    is `{name, level, hp, max_hp, status}`, read from row pairs (2i, 2i+1).
    It returns None unless the bottom box reads "Choose a POK…" or
    "Use item on" and at least one row parses.
  - `bag_screen(frame, classes, glyphs) -> list[str] | None`: item names
    down to CANCEL.
- **Tests:**
  - a synthetic party screen built from glyphs parses;
  - window 2's real frame gives 6 entries, with the levels
    26 27 28 26 26 25 and Pidgeotto PAR;
  - window 3's bag frame contains "BICYCLE".

### Task 3: Write party and bag — `poke/party.py`, `tests/test_party.py`

- **Interface:**
  - `species_by_name(rom, symbols)` and `items_by_name(rom, symbols)`;
  - `set_party(pyboy, rom, symbols, entries) -> dict`. It sets count,
    species list, structs and nicknames, OT name and ID = the player's.
    Types come from BaseStats, moves are the 4 starting moves, and the stats
    use the Gen-1 formula with DV 8. It returns the guessed fields.
  - `set_bag(pyboy, rom, symbols, names, quantity=1)`.
- **Test** (skips without the ROM): set a 3-mon party on a colour state,
  open the party menu in the emulator (START, down, A), and read it back with
  Task 2. The entries are the same.

### Task 4: Apply at window start; menu-cursor tier; budget per wall — `scripts/repair_labels.py`

- `--rebuild`:
  - after anchoring, scan the window's video every 4 frames for the first
    `party_screen` that parses the same on two frames 4 apart, and the first
    `bag_screen`;
  - apply them with `set_party` / `set_bag`, and list what was set in the
    report under `rebuilt`.
- `--resync` accepts `menu` (MENU_CURSOR, a cheap tier).
- **The budget belongs to a wall.** The deadline resets only when the loss
  moves past the furthest loss reached so far.
- **Tests:**
  - the budget test: a backtrack that re-finds the same wall gets no new
    budget;
  - `--resync menu` parses.

### Task 5: Re-run the three windows

- Same anchors as before:
  `--rebuild --resync tile,menu --dry-run --stuck-budget 900`.
- Render each window's new wall.
- Report the frames in sync before and after, and what stopped each one.
