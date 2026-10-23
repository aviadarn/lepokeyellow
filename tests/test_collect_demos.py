"""Building a behaviour-cloning dataset from the oracle's solution."""

import numpy as np
import pytest

from scripts.collect_demos import PHASE_OBJECTIVE, demo_steps


class ReplayEnv:
    """Snapshot/restore over a counter, so a test can check which state each
    frame was taken from."""

    def __init__(self):
        self.value = 0
        self.played = []

    def snapshot(self):
        return self.value

    def restore(self, blob):
        self.value = blob

    def screen(self):
        return np.full((144, 160, 3), self.value, dtype=np.uint8)

    def step(self, action_index):
        self.played.append(action_index)
        self.value += 1


def test_each_frame_comes_from_the_state_its_button_was_chosen_from():
    env = ReplayEnv()
    steps = demo_steps(env, edge_blobs=[10, 20], edge_traces=[["a"], ["up", "a"]])
    assert [s.action_name for s in steps] == ["a", "up", "a"]
    # Edge 0 starts at 10; edge 1 is restored to 20 rather than continuing at 11.
    assert [int(s.frame[0, 0]) for s in steps] == [10, 20, 21]


def test_the_trace_is_flattened_in_order():
    env = ReplayEnv()
    steps = demo_steps(env, edge_blobs=[0, 0], edge_traces=[["a", "a"], ["down"]])
    assert len(steps) == 3


def test_an_unsolved_phase_contributes_nothing():
    env = ReplayEnv()
    assert demo_steps(env, edge_blobs=[], edge_traces=[]) == []


def test_frames_are_greyscale_two_dimensional():
    env = ReplayEnv()
    steps = demo_steps(env, edge_blobs=[3], edge_traces=[["a"]])
    assert steps[0].frame.shape == (144, 160)
    assert steps[0].frame.dtype == np.uint8


def test_every_phase_names_the_objective_it_demonstrates():
    """The clone is conditioned on the objective, as the RL policy is."""
    from poke.objectives import CHAIN

    keys = {objective.key for objective in CHAIN}
    for phase, objective in PHASE_OBJECTIVE.items():
        assert objective in keys, f"{phase} -> {objective} is not an objective"


class BoxEnv(ReplayEnv):
    """Control is free on even counter values, a box is drawn on odd ones."""

    def screen(self):
        frame = np.zeros((144, 160, 3), dtype=np.uint8)
        if self.value % 2:
            frame[96:144, :, 0] = 255          # white paper, two shades
        else:
            frame[96:144, :, 0] = 255
            frame[96:144, ::3, 0] = 170        # scenery: four shades
            frame[96:144, ::5, 0] = 85
            frame[96:144, ::7, 0] = 0
        return frame


def test_each_step_records_whether_the_policy_would_have_owned_the_turn():
    """Under the contract the harness owns text and battles. A clone trained
    on those learns to mash A, which is the degenerate behaviour we are
    trying to get away from."""
    env = BoxEnv()
    steps = demo_steps(env, edge_blobs=[0, 1], edge_traces=[["up"], ["a"]])
    assert [s.policy_turn for s in steps] == [True, False]


def test_policy_turns_filters_the_harness_work_out():
    from scripts.collect_demos import policy_turns

    env = BoxEnv()
    steps = demo_steps(env, edge_blobs=[0, 1], edge_traces=[["up"], ["a"]])
    kept = policy_turns(steps)
    assert len(kept) == 1 and kept[0].action_name == "up"
