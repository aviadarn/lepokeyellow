"""The story, read out of RAM: event flags and the bag.

`poke.ram` covers the things a frame-by-frame controller needs (position,
party, battle). This module covers the things the *plot* is recorded in, which
is what decides which objective the player is on.

Bit indices follow pret/pokeyellow's event_constants. Bits 37, 56 and 57 were
watched flipping on this cartridge, in order, during the parcel errand. The
catch-training bits (45, 47) fire after the Pokedex, and are listed here for
the objectives that come later rather than because this errand set them.
"""

from __future__ import annotations

GOT_STARTER = 34
BATTLED_RIVAL_IN_OAKS_LAB = 35
GOT_POKEBALLS_FROM_OAK = 36
GOT_POKEDEX = 37
OAK_APPEARED_IN_PALLET = 39
SPAWNED_OLD_MAN_1 = 44
COMPLETED_CATCH_TRAINING = 45
INITIAL_CATCH_TRAINING = 47
OAK_GOT_PARCEL = 56
GOT_OAKS_PARCEL = 57

NUM_EVENT_BYTES = 320
OAKS_PARCEL = 0x46
BAG_CAPACITY = 20


#: The bit indices below are Yellow's. poke.games exists so a Yellow ROM
#: cannot be read with Red's addresses; the indices need the same guard, or
#: pointing this at Red would read a correctly-shifted byte and report the
#: wrong story.
SUPPORTED_GAMES = frozenset({"yellow"})


def _check_game(env) -> None:
    if env.game.key not in SUPPORTED_GAMES:
        raise RuntimeError(
            f"poke.flags holds Yellow's event bit indices; {env.game.key!r} "
            "has its own. Confirm them against that ROM before using them."
        )


def event_bit(env, index: int) -> int:
    """One bit of wEventFlags. An unreadable byte reads as 0, not as a crash:
    a missing read means "this has not happened", which is the safe answer."""
    _check_game(env)
    byte = env.read_ram(env.game.ram.event_flags + index // 8)
    return 0 if byte is None else (byte >> (index % 8)) & 1


def bag_items(env) -> list[tuple[int, int]]:
    """(item id, quantity) pairs. The count byte is trusted only up to the
    bag's real capacity -- a wrong address reads a huge count and would
    otherwise march off through WRAM inventing items."""
    base = env.game.ram.num_bag_items
    count = env.read_ram(base)
    if count is None or count > BAG_CAPACITY:
        return []
    pairs = []
    for slot in range(count):
        item = env.read_ram(base + 1 + 2 * slot)
        qty = env.read_ram(base + 2 + 2 * slot)
        if item is None:
            break
        pairs.append((int(item), 0 if qty is None else int(qty)))
    return pairs


def carrying_parcel(env) -> bool:
    """Whether Oak's Parcel is in the bag right now.

    Not `event_bit(env, GOT_OAKS_PARCEL)`: measured on the cartridge, bit 57
    never clears when Oak takes the parcel. The bit says "was collected"; the
    bag says "is held".
    """
    _check_game(env)
    return any(item == OAKS_PARCEL for item, _ in bag_items(env))
