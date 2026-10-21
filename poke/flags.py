"""The story, read out of RAM: event flags and the bag.

`poke.ram` covers the things a frame-by-frame controller needs (position,
party, battle). This module covers the things the *plot* is recorded in, which
is what decides which objective the player is on.

Bit indices follow pret/pokeyellow's event_constants. The ones below were
watched flipping on this cartridge, in order, during the parcel errand.
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


def event_bit(env, index: int) -> int:
    """One bit of wEventFlags. An unreadable byte reads as 0, not as a crash:
    a missing read means "this has not happened", which is the safe answer."""
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
    return any(item == OAKS_PARCEL for item, _ in bag_items(env))
