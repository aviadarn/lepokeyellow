"""Game-aware RAM reads.

Replaces the module-level constants in stmonty/lePokeRed with lookups through
the env's detected `Game`, so a Yellow ROM cannot be read with Red addresses.
"""

from __future__ import annotations

import numpy as np

from poke.games import EMPTY_SPECIES, PARTY_MON_BYTES


def party_count(env) -> int:
    value = env.read_ram(env.game.ram.party_count)
    return 0 if value is None else int(value)


def party_species(env) -> int | None:
    species = env.read_ram(env.game.ram.party_species)
    return None if species is None or species == EMPTY_SPECIES else int(species)


def cur_map(env) -> int | None:
    value = env.read_ram(env.game.ram.cur_map)
    return None if value is None else int(value)


def coords(env) -> tuple[int | None, int | None]:
    """Player tile (y, x). Confirmed on the cartridge, not just the shift rule."""
    return env.read_ram(env.game.ram.y_coord), env.read_ram(env.game.ram.x_coord)


# wIsInBattle. Confirmed on the cartridge: it read 2 through the forced rival
# fight in Oak's lab and cleared to 0 the moment the fight ended.
NO_BATTLE, WILD_BATTLE, TRAINER_BATTLE = 0, 1, 2

BATTLE_NAMES = {
    NO_BATTLE: "none",
    WILD_BATTLE: "wild",
    TRAINER_BATTLE: "trainer",
}


def battle_kind(env) -> int:
    value = env.read_ram(env.game.ram.in_battle)
    return NO_BATTLE if value is None else int(value)


def in_battle(env) -> bool:
    """True while a battle is running.

    The planner has no business steering a battle menu -- it is a different
    dynamics problem from walking, and unavoidable: the rival challenges the
    player the moment Oak hands over Pikachu. So a scripted policy takes over
    while this is set. Mashing A is enough for that first fight: 43 A presses
    cleared it with Pikachu intact.
    """
    return battle_kind(env) != NO_BATTLE


def has_pokemon(env) -> bool:
    """True once the party holds at least one Pokemon.

    Party count is a 0..6 counter in a byte that is 0 on a fresh file, so a
    read of exactly 1..6 is the signal. Values above 6 mean the address is
    wrong for this ROM, not that the player has seven Pokemon.
    """
    count = party_count(env)
    if count > 6:
        raise RuntimeError(
            f"party_count read {count} at {env.game.ram.party_count:#06x} for "
            f"{env.game.key}; address is wrong for this ROM. "
            "Run scripts/calibrate_ram.py."
        )
    return count >= 1


# wTileMap: the 20x18 tile buffer the game copies to the screen. Below
# $CF1A, so the same address in Red and Yellow (pret sym agrees).
TILE_MAP = 0xC3A0
TILE_MAP_ROWS, TILE_MAP_COLS = 18, 20

# Gen-1 box-border tiles, top-left and bottom-right corners.
BOX_TOP_LEFT, BOX_BOTTOM_RIGHT = 0x79, 0x7E


def tile_map(env) -> np.ndarray:
    raw = env.read_ram_range(TILE_MAP, TILE_MAP + TILE_MAP_ROWS * TILE_MAP_COLS)
    return np.frombuffer(raw, dtype=np.uint8).reshape(TILE_MAP_ROWS, TILE_MAP_COLS)


def dialog_open(env) -> bool:
    """True while the standard message box is drawn across the bottom.

    Read from the tile buffer, not the pixels, so it holds in colour, where
    `poke.control.text_box_open`'s DMG shade test does not. The message box is
    drawn at tile (0, 12) to (19, 17), and on 3,000 random colour steps from
    the bedroom the two corner tiles separated box from no box in every
    frame inspected. `wFontLoaded` disagreed on 11 transition frames, so it is
    not used. In a battle the bottom box is always up, so this reads True there.
    """
    tiles = tile_map(env)
    return bool(tiles[12, 0] == BOX_TOP_LEFT and tiles[17, 19] == BOX_BOTTOM_RIGHT)


def facing(env) -> int:
    """0 down, 4 up, 8 left, 12 right: the sprite engine's own values, which are
    also PokeDreamer's `facings` encoding."""
    value = env.read_ram(env.game.ram.facing)
    return 0 if value is None else int(value)


def badge_count(env) -> int:
    value = env.read_ram(env.game.ram.badges)
    return 0 if value is None else int(value).bit_count()


def _u16(env, addr: int) -> int:
    return (env.read_ram(addr) << 8) | env.read_ram(addr + 1)


def party_hp(env) -> tuple[np.ndarray, np.ndarray]:
    """(current, max) HP for six slots, 0 past the party's end."""
    ram = env.game.ram
    count = min(party_count(env), 6)
    hp = np.zeros(6, dtype=np.int16)
    max_hp = np.zeros(6, dtype=np.int16)
    for slot in range(count):
        offset = slot * PARTY_MON_BYTES
        hp[slot] = _u16(env, ram.party_hp + offset)
        max_hp[slot] = _u16(env, ram.party_max_hp + offset)
    return hp, max_hp


def party_levels(env) -> np.ndarray:
    count = min(party_count(env), 6)
    return np.array([env.read_ram(env.game.ram.party_level + k * PARTY_MON_BYTES)
                     for k in range(count)] + [0] * (6 - count), dtype=np.uint8)
