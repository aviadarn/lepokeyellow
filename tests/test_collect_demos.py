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
