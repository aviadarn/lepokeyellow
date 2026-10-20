import numpy as np
import pytest
import torch as t

from poke.probe_mlp import MLPHead, apply_stack, fit_mlp, stack


def linear_problem(n=600, dim=12, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, dim))
    direction = rng.normal(size=dim)
    return x, x @ direction + 3.0


def test_fits_a_function_a_linear_probe_could_also_fit():
    x, y = linear_problem()
    head = fit_mlp(x, y, epochs=200, patience=40, seed=0)
    error = np.abs(head.predict(x) - y).mean()
    assert error < 0.2 * np.abs(y - y.mean()).mean()


def test_fits_a_function_a_linear_probe_cannot():
    """The reason for the head at all: a linear read-out of the encoder put the
    player 1.4-1.9 tiles out in Pallet Town while the pixels said the
    information was there."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(800, 6))
    y = np.abs(x[:, 0]) * 3.0 + np.sin(x[:, 1] * 2.0)
    head = fit_mlp(x, y, epochs=300, patience=60, seed=0)
    mlp_error = np.abs(head.predict(x) - y).mean()

    centred = x - x.mean(axis=0)
    weights = np.linalg.solve(centred.T @ centred + np.eye(6), centred.T @ (y - y.mean()))
    linear_error = np.abs(centred @ weights + y.mean() - y).mean()
    assert mlp_error < 0.5 * linear_error


def test_numpy_and_torch_evaluate_the_same_weights_identically():
    """The head is fitted and measured through numpy and used for planning
    through torch, so the two paths have to agree -- hence ReLU rather than a
    GELU approximation that differs between them."""
    x, y = linear_problem(dim=8)
    head = fit_mlp(x, y, epochs=50, patience=10, seed=0)
    stacked = stack([head, head], t.device("cpu"))
    through_torch = apply_stack(t.from_numpy(x).float(), stacked).numpy()
    through_numpy = head.predict(x)
    assert np.allclose(through_torch[:, 0], through_numpy, atol=1e-4)
    assert np.allclose(through_torch[:, 1], through_numpy, atol=1e-4)


def test_shuffled_labels_do_not_fit():
    """The control that matters for a head with the capacity to memorise."""
    x, y = linear_problem()
    rng = np.random.default_rng(2)
    shuffled = y.copy()
    rng.shuffle(shuffled)
    head = fit_mlp(x, shuffled, epochs=200, patience=40, seed=0)
    assert head.val_mae > 0.7 * np.abs(shuffled - shuffled.mean()).mean()


def test_early_stopping_holds_out_whole_groups():
    """Frames 60 emulator frames apart are near-identical, so a validation
    split that cuts inside an episode is reading leaked neighbours."""
    x, y = linear_problem(n=400)
    groups = np.arange(400) // 100
    head = fit_mlp(x, y, groups=groups, epochs=30, patience=10, seed=0)
    assert head.epochs_used >= 1


def test_target_standardisation_is_folded_into_the_output_layer():
    """Nothing downstream should have to carry the target's mean and scale."""
    x, y = linear_problem()
    head = fit_mlp(x, y + 1000.0, epochs=100, patience=20, seed=0)
    assert np.abs(head.predict(x).mean() - (y.mean() + 1000.0)) < 20.0


def test_stack_rejects_heads_of_different_shapes():
    x, y = linear_problem()
    wide = fit_mlp(x, y, hidden=16, epochs=5, seed=0)
    narrow = fit_mlp(x, y, hidden=8, epochs=5, seed=0)
    with pytest.raises(ValueError, match="mismatched shapes"):
        stack([wide, narrow], t.device("cpu"))


def test_stack_needs_at_least_one_head():
    with pytest.raises(ValueError, match="at least one head"):
        stack([], t.device("cpu"))


def test_mismatched_lengths_are_rejected():
    x, y = linear_problem()
    with pytest.raises(ValueError, match="same length"):
        fit_mlp(x, y[:-1])


def test_predict_is_pure_affine_when_depth_is_one():
    x, y = linear_problem(dim=4)
    head = fit_mlp(x, y, depth=1, hidden=5, epochs=20, seed=0)
    assert len(head.weights) == 2
    doubled = head.predict(np.concatenate([x, x]))
    assert np.allclose(doubled[: len(x)], doubled[len(x) :])
