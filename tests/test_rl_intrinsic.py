"""The world-model exploration bonus. Training only, policy steps only."""

import numpy as np
import pytest
import torch

from poke.rl.intrinsic import CHOSEN, CLIP, WARMUP, RunningNorm, WorldModelBonus


class FakePredictor:
    """Stands in for the JEPA: predicts whatever it was told to."""

    def __init__(self, prediction):
        self.prediction = torch.tensor(prediction, dtype=torch.float32)
        self.calls = 0

    def rollout(self, initial, actions):
        self.calls += 1
        return self.prediction.view(1, 1, -1)


def test_running_norm_tracks_mean_and_spread():
    norm = RunningNorm()
    for value in [1.0, 2.0, 3.0, 4.0]:
        norm.update(value)
    assert norm.mean == pytest.approx(2.5)
    assert norm.std == pytest.approx(np.std([1, 2, 3, 4], ddof=0), rel=1e-3)


def test_running_norm_divides_by_the_spread():
    norm = RunningNorm()
    for value in [0.0, 2.0]:
        norm.update(value)
    assert norm.normalise(2.0) == pytest.approx(2.0 / (1.0 + 1e-8), rel=1e-6)


def warm(bonus, n=WARMUP):
    """Spend the warm-up, with a spread to divide by rather than a constant."""
    for i in range(n):
        bonus.bonus(np.zeros(2, dtype=np.float32), 0,
                    np.array([float(i % 3), 0.0], dtype=np.float32))


def test_the_first_transitions_pay_nothing_while_the_scale_is_unknown():
    """With one sample the running std is 0 and dividing by the 1e-8 guard
    paid 2,500,000 for a single frame."""
    bonus = WorldModelBonus(FakePredictor([0.0, 0.0]), device="cpu", coef=1.0)
    assert bonus.bonus(np.zeros(2, dtype=np.float32), 0,
                       np.array([5.0, 0.0], dtype=np.float32)) == 0.0


def test_one_outlier_is_worth_less_than_a_milestone():
    bonus = WorldModelBonus(FakePredictor([0.0, 0.0]), device="cpu", coef=0.05)
    warm(bonus)
    huge = bonus.bonus(np.zeros(2, dtype=np.float32), 0,
                       np.array([1e6, 0.0], dtype=np.float32))
    assert huge == pytest.approx(0.05 * CLIP)
    assert huge < 1.0, "an objective pays 1.0 and must stay the bigger prize"


def test_a_surprising_transition_scores_higher_than_a_predicted_one():
    predictor = FakePredictor([1.0, 0.0])
    bonus = WorldModelBonus(predictor, device="cpu", coef=1.0)
    warm(bonus)
    predicted = bonus.bonus(np.zeros(2, dtype=np.float32), 1, np.array([1.0, 0.0], dtype=np.float32))
    surprising = bonus.bonus(np.zeros(2, dtype=np.float32), 1, np.array([0.0, 9.0], dtype=np.float32))
    assert surprising > predicted


def test_the_bonus_is_scaled_by_its_own_running_spread():
    """Raw latent MSE has no natural scale; the extrinsic reward does."""
    predictor = FakePredictor([0.0, 0.0])
    bonus = WorldModelBonus(predictor, device="cpu", coef=0.05)
    warm(bonus)
    values = [bonus.bonus(np.zeros(2, dtype=np.float32), 0,
                          np.array([float(i), 0.0], dtype=np.float32)) for i in range(1, 30)]
    assert max(values) < 1.0, "a bonus must not dwarf the +1 an objective pays"


def test_the_coefficient_scales_the_output():
    predictor = FakePredictor([0.0, 0.0])
    loud = WorldModelBonus(predictor, device="cpu", coef=1.0)
    quiet = WorldModelBonus(predictor, device="cpu", coef=0.1)
    warm(loud)
    warm(quiet)
    target = np.array([3.0, 0.0], dtype=np.float32)
    for _ in range(5):
        loud_value = loud.bonus(np.zeros(2, dtype=np.float32), 0, target)
        quiet_value = quiet.bonus(np.zeros(2, dtype=np.float32), 0, target)
    assert loud_value == pytest.approx(quiet_value * 10.0, rel=1e-5)


def test_the_module_admits_the_choice_was_not_measured():
    """A tuned constant with no number beside it is a magic number. So is an
    unmeasured choice between two signals -- say so in the code."""
    assert CHOSEN["signal"] == "prediction_error"
    assert CHOSEN["measured"] is False
    assert "comparison" in CHOSEN["why"]


def test_bonus_many_matches_calling_bonus_one_at_a_time():
    """The driver pays four envs per step; four separate predictor forwards on
    MPS cost more than the emulators do."""
    prev = np.zeros((3, 2), dtype=np.float32)
    nexts = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float32)
    one_by_one = WorldModelBonus(FakePredictor([0.0, 0.0]), device="cpu", coef=1.0)
    batched = WorldModelBonus(FakePredictor([0.0, 0.0]), device="cpu", coef=1.0)
    warm(one_by_one)
    warm(batched)
    singles = [one_by_one.bonus(prev[i], 0, nexts[i]) for i in range(3)]
    many = batched.bonus_many(prev, [0, 0, 0], nexts)
    assert many == pytest.approx(singles)


def test_bonus_many_calls_the_predictor_once():
    predictor = FakePredictor([0.0, 0.0])
    bonus = WorldModelBonus(predictor, device="cpu", coef=1.0)
    before = predictor.calls
    bonus.bonus_many(np.zeros((4, 2), dtype=np.float32), [0, 1, 2, 3],
                     np.zeros((4, 2), dtype=np.float32))
    assert predictor.calls - before == 1
