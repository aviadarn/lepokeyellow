"""One emulator, the handlers, and the reward, as a step/reset loop."""

import numpy as np

from poke.actions import ACTIONS
from poke.flags import (
    BATTLED_RIVAL_IN_OAKS_LAB, GOT_OAKS_PARCEL, GOT_POKEDEX, OAK_APPEARED_IN_PALLET,
    OAK_GOT_PARCEL,
)
from poke.games import YELLOW
from poke.rl.trainenv import TrainEnv

A = ACTIONS.index("a")
UP = ACTIONS.index("up")


class FakeEmu:
    """A scriptable stand-in for PokeEnv. No ROM, no PyBoy."""

    def __init__(self, tile=(12, 35, 10), box=False, battle=0):
        self.game = YELLOW
        self.tile = tile
        self.box = box
        self.battle = battle
        self.ram_values = {}
        self.actions = []
        self.restored = None
        for bit in (OAK_APPEARED_IN_PALLET, BATTLED_RIVAL_IN_OAKS_LAB):
            self.set_bit(bit, 1)
        #: called after every step, with the env, so a test can script the world
        self.on_step = lambda env: None

    def set_bit(self, index, value=1):
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
        if addr == YELLOW.ram.in_battle:
            return self.battle
        if addr == YELLOW.ram.num_bag_items:
            return 0
        return self.ram_values.get(addr, 0)

    def screen(self):
        frame = np.zeros((144, 160, 3), dtype=np.uint8)
        if self.box:
            frame[96:144, :, :] = 255
        return frame

    def reset(self):
        return self.screen()

    def step(self, action_index):
        self.actions.append(action_index)
        self.on_step(self)
        return self.screen()

    def snapshot(self):
        return b"blob"

    def restore(self, blob):
        self.restored = blob


def test_a_policy_action_reaches_the_emulator():
    env = TrainEnv(env=FakeEmu(), episode_len=8)
    env.reset()
    step = env.step(UP)
    assert env.env.actions == [UP]
    assert step.info["controllers"] == {"policy": 1}


def test_the_frame_and_objective_come_back_shaped_for_the_encoder():
    env = TrainEnv(env=FakeEmu(), episode_len=8)
    step = env.reset()
    assert step.frame.shape == (144, 160)
    assert step.frame.dtype == np.uint8
    assert step.objective == [0.0, 0.0, 1.0, 0.0, 0.0], "get_parcel on Route 1"


def test_handler_turns_are_taken_inside_one_policy_step():
    emu = FakeEmu()
    emu.box = True
    env = TrainEnv(env=emu, episode_len=64, max_handler_steps=4)
    env.reset()
    before = len(emu.actions)
    step = env.step(UP)
    assert len(emu.actions) - before == 1 + 4, "the policy's action, then handler turns"
    assert step.info["handler_steps"] == 4, "this step's expansion"
    # Cumulative over the episode, and reset() drains too: 4 there, 4 here.
    assert step.info["controllers"]["text"] == 8


def test_reset_expands_handlers_before_first_obs():
    """Review focus 3: an archive state captured mid-dialogue."""
    emu = FakeEmu(box=True)

    def clear_the_box(env):
        if len(env.actions) >= 2:
            env.box = False

    emu.on_step = clear_the_box
    env = TrainEnv(env=emu, episode_len=64)
    step = env.reset()
    assert emu.box is False, "the box was cleared before the policy was asked"
    assert step.info["handler_steps"] >= 2


def test_handler_expansion_is_capped():
    env = TrainEnv(env=FakeEmu(box=True), episode_len=1000, max_handler_steps=3)
    env.reset()
    before = len(env.env.actions)
    env.step(UP)
    assert len(env.env.actions) - before == 4


def test_the_episode_ends_on_the_budget_and_handler_turns_count():
    emu = FakeEmu(box=True)
    env = TrainEnv(env=emu, episode_len=5, max_handler_steps=2)
    env.reset()
    step = env.step(UP)
    assert step.done is True, "1 policy action + 2 handler turns, twice, is over 5"
    assert step.info["taken"] >= 5


def test_the_episode_ends_when_the_forest_is_reached():
    emu = FakeEmu()
    env = TrainEnv(env=emu, episode_len=1000)
    env.reset()

    def enter_forest(inner):
        inner.set_bit(GOT_OAKS_PARCEL, 1)
        inner.set_bit(OAK_GOT_PARCEL, 1)
        inner.set_bit(GOT_POKEDEX, 1)
        inner.tile = (51, 47, 16)

    emu.on_step = enter_forest
    step = env.step(UP)
    assert step.done is True
    assert "enter_forest" in step.info["completed"]


def test_blackout_does_not_end_the_episode():
    """Review focus 1: the design refuses to special-case a faint."""
    emu = FakeEmu()
    env = TrainEnv(env=emu, episode_len=100)
    env.reset()
    emu.on_step = lambda inner: setattr(inner, "tile", (38, 7, 4))
    step = env.step(UP)
    assert step.done is False
    assert step.info["completed"] == []


def test_reset_from_a_blob_restores_it():
    emu = FakeEmu()
    env = TrainEnv(env=emu, episode_len=8)
    env.reset(blob=b"somewhere else")
    assert emu.restored == b"somewhere else"


def test_a_battle_is_the_handlers_and_not_the_policys():
    emu = FakeEmu(battle=1)
    env = TrainEnv(env=emu, episode_len=16, max_handler_steps=2)
    env.reset()
    step = env.step(UP)
    assert step.info["handler_steps"] == 2, "capped at max_handler_steps"
    assert step.info["controllers"]["battle"] == 4, "2 at reset, 2 after the policy action"
    assert step.info["controllers"]["policy"] == 1
