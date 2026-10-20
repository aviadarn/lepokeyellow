import pytest
import torch as t

from poke.jepa import JEPA
from poke.rollout import rollout_from_context


@pytest.fixture
def model():
    t.manual_seed(0)
    jepa = JEPA(num_frames=4)
    jepa.eval()
    return jepa


def test_one_frame_of_context_matches_the_planners_own_rollout(model):
    """With C=1 this has to be exactly JEPA.predict_rollout, or the measurement
    comparing context lengths is comparing two different things."""
    initial = t.randn(2, model.dim)
    actions = t.zeros(2, 5, 6)
    actions[:, :, 0] = 1.0
    with t.inference_mode():
        mine = rollout_from_context(model, initial[:, None, :], actions)
        theirs = model.rollout(initial, actions)
    assert t.allclose(mine, theirs, atol=1e-5)


def test_more_context_consumes_the_actions_already_taken(model):
    """Seeding with C real states needs the C-1 actions that produced them, so
    the predictor sees each state paired with the action taken AT it."""
    states = t.randn(2, 3, model.dim)
    actions = t.zeros(2, 3 + 4 - 1, 6)
    with t.inference_mode():
        out = rollout_from_context(model, states, actions)
    assert out.shape == (2, 4, model.dim)


def test_too_few_actions_is_a_clear_error(model):
    states = t.randn(1, 3, model.dim)
    with pytest.raises(ValueError, match="no horizon"):
        rollout_from_context(model, states, t.zeros(1, 2, 6))


def test_states_must_carry_a_context_axis(model):
    with pytest.raises(ValueError, match=r"\[B, C, D\]"):
        rollout_from_context(model, t.randn(2, model.dim), t.zeros(2, 3, 6))


def test_context_is_capped_by_the_predictors_positions(model):
    """A rollout longer than the predictor's position table must still run; it
    just stops growing its context."""
    initial = t.randn(1, 1, model.dim)
    horizon = model.predictor.pos.size(1) + 4
    with t.inference_mode():
        out = rollout_from_context(model, initial, t.zeros(1, horizon, 6))
    assert out.shape == (1, horizon, model.dim)
