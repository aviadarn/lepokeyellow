import pytest

from poke.flags import (
    BATTLED_RIVAL_IN_OAKS_LAB, GOT_OAKS_PARCEL, GOT_POKEDEX,
    OAKS_PARCEL, OAK_APPEARED_IN_PALLET, OAK_GOT_PARCEL,
)
from poke.games import YELLOW
from poke.objectives import CHAIN, N_OBJECTIVES, current, get, one_hot
from tests.conftest import FakeEnv

VIRIDIAN_FOREST = 51


class RecordingEnv(FakeEnv):
    """FakeEnv that remembers every address read, so a test can prove that a
    predicate never looked at the player's position."""

    def __init__(self, game, ram=None):
        super().__init__(game, ram)
        self.reads = []

    def read_ram(self, addr):
        self.reads.append(addr)
        return super().read_ram(addr)


def env_with(flags=(), bag=(), map_id=None):
    ram = {}
    for index in flags:
        addr = YELLOW.ram.event_flags + index // 8
        ram[addr] = ram.get(addr, 0) | (1 << (index % 8))
    ram[YELLOW.ram.num_bag_items] = len(bag)
    for slot, (item, qty) in enumerate(bag):
        ram[YELLOW.ram.num_bag_items + 1 + 2 * slot] = item
        ram[YELLOW.ram.num_bag_items + 2 + 2 * slot] = qty
    if map_id is not None:
        ram[YELLOW.ram.cur_map] = map_id
    return RecordingEnv(YELLOW, ram)


def test_chain_is_the_specs_order():
    assert [o.key for o in CHAIN] == [
        "trigger_oak", "get_starter", "get_parcel",
        "deliver_parcel", "enter_forest",
    ]
    assert [o.index for o in CHAIN] == [0, 1, 2, 3, 4]
    assert N_OBJECTIVES == 5


def test_new_game_is_on_trigger_oak():
    assert current(env_with()).key == "trigger_oak"


def test_oak_appeared_moves_to_the_starter():
    assert current(env_with(flags=[OAK_APPEARED_IN_PALLET])).key == "get_starter"


def test_rival_beaten_moves_to_the_parcel_errand():
    env = env_with(flags=[OAK_APPEARED_IN_PALLET, BATTLED_RIVAL_IN_OAKS_LAB])
    assert current(env).key == "get_parcel"


def test_parcel_in_hand_moves_to_delivery():
    env = env_with(
        flags=[OAK_APPEARED_IN_PALLET, BATTLED_RIVAL_IN_OAKS_LAB, GOT_OAKS_PARCEL],
        bag=[(OAKS_PARCEL, 1)],
    )
    assert current(env).key == "deliver_parcel"


def test_delivery_advances_even_though_bit_57_stays_set():
    """Bit 57 is still 1 after Oak takes the parcel. If `deliver_parcel` keyed
    off 57 the run would sit on it forever and never aim at the forest."""
    env = env_with(
        flags=[OAK_APPEARED_IN_PALLET, BATTLED_RIVAL_IN_OAKS_LAB,
               GOT_OAKS_PARCEL, OAK_GOT_PARCEL, GOT_POKEDEX],
        bag=[],
    )
    assert current(env).key == "enter_forest"


def test_the_policys_objective_never_reads_position():
    """`active` is a policy input, so it may read flags and the bag only."""
    forbidden = {YELLOW.ram.cur_map, YELLOW.ram.y_coord, YELLOW.ram.x_coord}
    for objective in CHAIN:
        env = env_with(map_id=VIRIDIAN_FOREST)
        objective.active(env)
        assert not (set(env.reads) & forbidden), objective.key


def test_reached_may_read_the_map_because_it_only_scores():
    env = env_with(map_id=VIRIDIAN_FOREST)
    assert get("enter_forest").reached(env)
    assert not get("enter_forest").reached(env_with(map_id=50))


def test_one_hot_is_the_policy_input():
    vector = one_hot(env_with(flags=[OAK_APPEARED_IN_PALLET]))
    assert vector == [0.0, 1.0, 0.0, 0.0, 0.0]
    assert len(vector) == N_OBJECTIVES


def test_unknown_objective_raises():
    with pytest.raises(ValueError, match="unknown objective"):
        get("catch_mewtwo")
