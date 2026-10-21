"""What the player is trying to do right now, read from the story flags.

The play-time contract lets the policy be told its current objective, and
nothing else about where it is. So the selector below reads event flags and
the bag and never the map or the coordinates: `active` is a policy input.
`reached` is allowed the map, because it scores a finished run rather than
steering a live one.

The chain runs from a new game to Viridian Forest. Leaving Pallet is not an
objective of its own -- it is folded into `get_parcel`, which is what makes the
Route 1 walk pay.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from poke.flags import (
    BATTLED_RIVAL_IN_OAKS_LAB, GOT_OAKS_PARCEL, GOT_POKEDEX,
    OAK_APPEARED_IN_PALLET, OAK_GOT_PARCEL, event_bit,
)
from poke.ram import cur_map

VIRIDIAN_FOREST = 51


@dataclass(frozen=True)
class Objective:
    key: str
    index: int
    description: str
    active: Callable[[object], bool]
    reached: Callable[[object], bool]


def _oak_not_yet(env):
    return event_bit(env, OAK_APPEARED_IN_PALLET) == 0


def _oak_appeared(env):
    return event_bit(env, OAK_APPEARED_IN_PALLET) == 1


def _rival_done(env):
    return event_bit(env, BATTLED_RIVAL_IN_OAKS_LAB) == 1


def _parcel_collected(env):
    return event_bit(env, GOT_OAKS_PARCEL) == 1


def _parcel_delivered(env):
    return event_bit(env, OAK_GOT_PARCEL) == 1


CHAIN: tuple[Objective, ...] = (
    Objective(
        key="trigger_oak",
        index=0,
        description="get Oak to appear and follow him into the lab",
        active=_oak_not_yet,
        reached=_oak_appeared,
    ),
    Objective(
        key="get_starter",
        index=1,
        description="take Pikachu and survive the rival fight",
        active=lambda env: _oak_appeared(env) and not _rival_done(env),
        reached=_rival_done,
    ),
    Objective(
        key="get_parcel",
        index=2,
        # Leaving Pallet and the whole of Route 1 live inside this one.
        description="cross Route 1 to Viridian and pick up Oak's Parcel",
        active=lambda env: _rival_done(env) and not _parcel_collected(env),
        reached=_parcel_collected,
    ),
    Objective(
        key="deliver_parcel",
        index=3,
        description="carry the parcel back to Oak and take the Pokedex",
        active=lambda env: _parcel_collected(env) and not _parcel_delivered(env),
        reached=lambda env: _parcel_delivered(env) and event_bit(env, GOT_POKEDEX) == 1,
    ),
    Objective(
        key="enter_forest",
        index=4,
        description="north past the old man into Viridian Forest",
        active=_parcel_delivered,
        reached=lambda env: cur_map(env) == VIRIDIAN_FOREST,
    ),
)

N_OBJECTIVES = len(CHAIN)


def current(env) -> Objective:
    """The first objective whose preconditions hold. Flags and bag only.

    The predicates are written to be mutually exclusive, but "first match"
    rather than "the one match" is deliberate: a save state captured mid-scene
    should pick the earliest plausible objective rather than raise.
    """
    for objective in CHAIN:
        if objective.active(env):
            return objective
    return CHAIN[-1]


def one_hot(env) -> list[float]:
    vector = [0.0] * N_OBJECTIVES
    vector[current(env).index] = 1.0
    return vector


def get(key: str) -> Objective:
    for objective in CHAIN:
        if objective.key == key:
            return objective
    known = ", ".join(o.key for o in CHAIN)
    raise ValueError(f"unknown objective {key!r}; known: {known}")
