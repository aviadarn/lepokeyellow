import numpy as np

from poke.actions import ACTIONS
from poke.games import YELLOW
from poke.oracle import search, story_key
from tests.conftest import FakeEnv

CORRIDOR = 6


class CorridorEnv(FakeEnv):
    """A one-room, one-axis world: x moves on left/right, walls at both ends.

    Enough to exercise the search's bookkeeping -- snapshots, dedup, the path
    it reconstructs -- without a cartridge.
    """

    def __init__(self):
        super().__init__(YELLOW, {
            YELLOW.ram.cur_map: 1,
            YELLOW.ram.y_coord: 0,
            YELLOW.ram.x_coord: 0,
            YELLOW.ram.in_battle: 0,
            YELLOW.ram.facing: 0,
            YELLOW.ram.event_flags: 0,
            YELLOW.ram.num_bag_items: 0,
        })
        self.steps = 0
        self.last = "noop"

    def snapshot(self):
        return dict(self.ram), self.last

    def restore(self, blob):
        self.ram, self.last = dict(blob[0]), blob[1]

    def screen(self):
        """The search's control probe presses `down` and then `right` from one
        snapshot and calls them control if the two screens differ. Encoding
        only x would make them identical at the wall, where neither moves, and
        the fake would report a cutscene it is not having -- so the last
        button is on screen too. Control is always free in the corridor."""
        frame = np.zeros((144, 160, 3), dtype=np.uint8)
        frame[0, 0, 0] = self.ram[YELLOW.ram.x_coord]
        frame[0, 1, 0] = ACTIONS.index(self.last)
        return frame

    def step(self, action_index):
        self.steps += 1
        button = ACTIONS[action_index]
        self.last = button
        x = self.ram[YELLOW.ram.x_coord]
        if button == "right" and x < CORRIDOR - 1:
            self.ram[YELLOW.ram.x_coord] = x + 1
        if button == "left" and x > 0:
            self.ram[YELLOW.ram.x_coord] = x - 1
        if button == "a" and x == CORRIDOR - 1:
            self.ram[YELLOW.ram.event_flags] = 1


def test_search_finds_the_shortest_route_down_the_corridor():
    env = CorridorEnv()
    result = search(
        env,
        goal=lambda: env.ram[YELLOW.ram.x_coord] == CORRIDOR - 1,
        allowed={1},
        buttons=("left", "right"),
        with_facing=False,
    )
    assert result.found
    assert result.path == ["right"] * (CORRIDOR - 1)
    assert result.end == (1, 0, CORRIDOR - 1)


def test_search_reports_failure_rather_than_hanging():
    env = CorridorEnv()
    result = search(
        env,
        goal=lambda: env.ram[YELLOW.ram.x_coord] == 99,
        allowed={1},
        buttons=("left", "right"),
        with_facing=False,
        max_nodes=50,
    )
    assert not result.found
    assert result.path == []
    assert result.nodes <= 50


def test_search_will_not_leave_the_allowed_maps():
    env = CorridorEnv()
    result = search(
        env,
        goal=lambda: False,
        allowed=set(),
        buttons=("right",),
        with_facing=False,
        max_nodes=20,
    )
    assert not result.found
    assert result.maps_seen == {}


def test_a_press_that_flips_a_flag_is_reachable():
    env = CorridorEnv()
    result = search(
        env,
        goal=lambda: env.ram[YELLOW.ram.event_flags] == 1,
        allowed={1},
        buttons=("right", "a"),
        with_facing=False,
    )
    assert result.found
    assert result.path[-1] == "a"


def test_the_result_carries_the_literal_actions_not_just_the_edges():
    """Each edge is one button plus however many A presses `transition` spent
    waiting for control. Replaying the edge buttons alone desynchronises on
    the first edge that waited, so the result carries both."""
    env = CorridorEnv()
    result = search(
        env,
        goal=lambda: env.ram[YELLOW.ram.event_flags] == 1,
        allowed={1},
        buttons=("right", "a"),
        with_facing=False,
    )
    assert result.found
    assert result.actions[: len(result.path)] or result.actions
    assert len(result.actions) >= len(result.path)
    assert all(button in ACTIONS for button in result.actions)
    assert result.actions[-1] == "a"


def test_story_key_separates_states_that_differ_only_in_flags():
    env = CorridorEnv()
    before = story_key(env, with_facing=False)
    env.ram[YELLOW.ram.event_flags] = 1
    assert story_key(env, with_facing=False) != before
