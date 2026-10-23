"""Recording demonstrations: the frame a decision was made from, and the
action taken from it."""

import numpy as np
import pytest

from poke.record import RecordingEnv


class StepEnv:
    """An env whose screen changes with every step, so a test can tell which
    frame was captured."""

    game = "yellow"

    def __init__(self):
        self.n = 0
        self.closed = False
        self.taken = []

    def screen(self):
        return np.full((144, 160, 3), self.n, dtype=np.uint8)

    def step(self, action_index):
        self.taken.append(action_index)
        self.n += 1
        return self.screen()

    def close(self):
        self.closed = True

    def whatever(self, x):
        return x * 2


def test_it_records_the_frame_the_action_was_chosen_from():
    """Not the frame after. Behaviour cloning learns frame -> action, and the
    frame after the action already contains the answer."""
    env = RecordingEnv(StepEnv())
    env.step(3)
    env.step(5)
    assert [step.action for step in env.steps] == [3, 5]
    assert env.steps[0].frame[0, 0] == 0, "the screen before the first step"
    assert env.steps[1].frame[0, 0] == 1


def test_the_frame_is_greyscale_and_two_dimensional():
    env = RecordingEnv(StepEnv())
    env.step(0)
    assert env.steps[0].frame.shape == (144, 160)
    assert env.steps[0].frame.dtype == np.uint8


def test_it_passes_the_step_through_untouched():
    inner = StepEnv()
    env = RecordingEnv(inner)
    returned = env.step(4)
    assert inner.taken == [4]
    assert returned[0, 0, 0] == 1


def test_unknown_attributes_reach_the_wrapped_env():
    """play.py touches a great many methods; the recorder must be invisible."""
    env = RecordingEnv(StepEnv())
    assert env.whatever(21) == 42
    env.close()
    assert env._env.closed


def test_a_label_can_be_attached_to_the_steps_that_follow():
    """The planner knows which milestone it is pursuing; the dataset should
    carry it, so a clone can be conditioned on the objective."""
    env = RecordingEnv(StepEnv())
    env.label("get_parcel")
    env.step(1)
    env.label("deliver_parcel")
    env.step(2)
    assert [step.label for step in env.steps] == ["get_parcel", "deliver_parcel"]


def test_recording_can_be_paused():
    """A search that rolls the emulator back must not pollute the dataset."""
    env = RecordingEnv(StepEnv())
    env.step(1)
    with env.paused():
        env.step(2)
        env.step(3)
    env.step(4)
    assert [step.action for step in env.steps] == [1, 4]
    assert env._env.taken == [1, 2, 3, 4], "the game still ran throughout"


def test_the_step_count_is_what_was_recorded_not_what_was_played():
    env = RecordingEnv(StepEnv())
    with env.paused():
        env.step(0)
    assert len(env) == 0
