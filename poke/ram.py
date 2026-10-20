"""Game-aware RAM reads.

Replaces the module-level constants in stmonty/lePokeRed with lookups through
the env's detected `Game`, so a Yellow ROM cannot be read with Red addresses.
"""

from __future__ import annotations

from poke.games import EMPTY_SPECIES


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


def in_battle(env) -> bool:
    """True while a battle is running.

    The planner has no business steering a battle menu, so `scripts/play.py`
    hands control to a scripted policy while this is set.
    """
    value = env.read_ram(env.game.ram.in_battle)
    return bool(value)


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
