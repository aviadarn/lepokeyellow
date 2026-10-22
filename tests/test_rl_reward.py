"""The training reward. It reads RAM on purpose: it is never a policy input."""

import pytest

from poke.flags import GOT_OAKS_PARCEL, OAK_GOT_PARCEL, GOT_POKEDEX
from poke.games import YELLOW
from poke.rl.reward import OBJECTIVE_REWARD, TILE_REWARD, RewardState, step_reward


class FakeEnv:
    """Enough env for flags, the bag and position."""

    def __init__(self, ram=None, bits=(), tile=(12, 35, 10)):
        self.game = YELLOW
        self.ram_values = dict(ram or {})
        self.tile = tile
        for bit in bits:
            self.set_bit(bit, 1)

    def set_bit(self, index, value):
        addr = YELLOW.ram.event_flags + index // 8
        byte = self.ram_values.get(addr, 0)
        mask = 1 << (index % 8)
        self.ram_values[addr] = (byte | mask) if value else (byte & ~mask)

    def read_ram(self, addr):
        if addr == YELLOW.ram.cur_map:
            return self.tile[0]
        if addr == YELLOW.ram.y_coord:
            return self.tile[1]
        if addr == YELLOW.ram.x_coord:
            return self.tile[2]
        if addr == YELLOW.ram.num_bag_items:
            return 0
        return self.ram_values.get(addr, 0)


def route1_env(**kwargs):
    """Stage A's start: the rival fight is done, nothing else is."""
    from poke.flags import BATTLED_RIVAL_IN_OAKS_LAB, OAK_APPEARED_IN_PALLET
    return FakeEnv(bits=(OAK_APPEARED_IN_PALLET, BATTLED_RIVAL_IN_OAKS_LAB), **kwargs)


def test_a_new_tile_pays_once():
    env = route1_env()
    state = RewardState.start(env)
    assert step_reward(env, state) == (TILE_REWARD, [])
    assert step_reward(env, state) == (0.0, [])


def test_completing_an_objective_pays_one():
    env = route1_env()
    state = RewardState.start(env)
    step_reward(env, state)
    env.set_bit(GOT_OAKS_PARCEL, 1)
    env.tile = (42, 5, 2)
    reward, completed = step_reward(env, state)
    assert completed == ["get_parcel"]
    assert reward == pytest.approx(OBJECTIVE_REWARD + TILE_REWARD)


def test_the_tile_set_resets_when_the_objective_changes():
    """The reset fires on the step the objective changes, so the test has to
    leave the tile first and come back under the new objective."""
    env = route1_env()
    state = RewardState.start(env)
    step_reward(env, state)             # (12, 35, 10) paid under get_parcel
    env.tile = (12, 36, 10)
    step_reward(env, state)
    env.set_bit(GOT_OAKS_PARCEL, 1)
    env.tile = (42, 5, 2)               # the objective flips here; tiles reset
    step_reward(env, state)
    env.tile = (12, 35, 10)             # the very first tile, new objective
    reward, _ = step_reward(env, state)
    assert reward == pytest.approx(TILE_REWARD), "the walk back is new ground"


def test_objectives_already_true_at_the_start_are_not_credited():
    env = route1_env()
    env.set_bit(GOT_OAKS_PARCEL, 1)
    state = RewardState.start(env)
    reward, completed = step_reward(env, state)
    assert completed == []
    assert reward == pytest.approx(TILE_REWARD)


def test_an_objective_is_credited_once_even_if_the_reading_regresses():
    """Review focus 4: `current()` can move backwards. Pay once anyway."""
    env = route1_env()
    state = RewardState.start(env)
    env.set_bit(GOT_OAKS_PARCEL, 1)
    first, completed = step_reward(env, state)
    assert completed == ["get_parcel"]
    env.set_bit(GOT_OAKS_PARCEL, 0)     # the reading goes backwards
    step_reward(env, state)
    env.set_bit(GOT_OAKS_PARCEL, 1)     # and comes back
    env.tile = (1, 20, 20)
    again, completed_again = step_reward(env, state)
    assert completed_again == []
    assert again == pytest.approx(TILE_REWARD)


def test_blackout_credits_no_objective():
    """Review focus 1: a faint warps the player home. Flags do not move."""
    env = route1_env()
    state = RewardState.start(env)
    step_reward(env, state)
    env.tile = (38, 7, 4)               # REDS_HOUSE_2F, where a blackout lands
    reward, completed = step_reward(env, state)
    assert completed == []
    assert reward == pytest.approx(TILE_REWARD), "one new tile, nothing else"


def test_the_pokedex_is_part_of_deliver_parcel():
    env = route1_env()
    env.set_bit(GOT_OAKS_PARCEL, 1)
    state = RewardState.start(env)
    env.set_bit(OAK_GOT_PARCEL, 1)
    reward, completed = step_reward(env, state)
    assert completed == [], "the parcel is not delivered until the Pokedex lands"
    env.set_bit(GOT_POKEDEX, 1)
    reward, completed = step_reward(env, state)
    assert completed == ["deliver_parcel"]
