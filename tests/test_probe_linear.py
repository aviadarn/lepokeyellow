import numpy as np
import pytest

from poke.probe_linear import blocked_folds, cross_val, fit_ridge, predict, r2


def test_ridge_recovers_a_linear_relationship():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 4))
    y = x @ np.array([1.0, -2.0, 0.5, 0.0]) + 3.0
    weights, intercept = fit_ridge(x, y, alpha=1e-8)
    assert np.allclose(weights, [1.0, -2.0, 0.5, 0.0], atol=1e-4)
    assert abs(intercept - 3.0) < 1e-4


def test_ridge_shrinks_with_alpha():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 3))
    y = x @ np.array([5.0, 5.0, 5.0])
    small, _ = fit_ridge(x, y, alpha=1e-8)
    large, _ = fit_ridge(x, y, alpha=1e4)
    assert np.abs(large).sum() < np.abs(small).sum()


def test_blocked_folds_cover_every_sample_exactly_once():
    folds = blocked_folds(100, folds=5, purge=0)
    covered = np.concatenate([test for _, test in folds])
    assert sorted(covered.tolist()) == list(range(100))


def test_purge_removes_neighbours_from_training():
    """The whole point: a test block's neighbours must not be trainable on, or
    the probe interpolates between two nearly identical frames."""
    train, test = blocked_folds(100, folds=5, purge=10)[2]
    assert test.min() == 40 and test.max() == 59
    assert not set(range(30, 70)) & set(train.tolist())
    assert 29 in train and 70 in train


def test_purge_at_the_edges_does_not_wrap():
    train, test = blocked_folds(100, folds=5, purge=10)[0]
    assert test.min() == 0
    assert train.min() == 30


def test_folds_must_be_at_least_two():
    with pytest.raises(ValueError, match="folds must be at least 2"):
        blocked_folds(10, folds=1)


def test_r2_is_one_for_perfect_and_zero_for_the_mean():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r2(y, y) == pytest.approx(1.0)
    assert r2(y, np.full_like(y, y.mean())) == pytest.approx(0.0)


def test_r2_goes_negative_when_worse_than_the_mean():
    y = np.array([1.0, 2.0, 3.0])
    assert r2(y, np.array([10.0, -10.0, 10.0])) < 0


def test_cross_val_scores_a_learnable_signal_high():
    rng = np.random.default_rng(0)
    y = np.linspace(0, 1, 300)
    x = np.stack([y, rng.normal(scale=0.01, size=300)], axis=1)
    assert cross_val(x, y, alpha=1e-6, folds=5, purge=5)["r2"] > 0.95


def test_cross_val_scores_pure_noise_at_or_below_zero():
    """Shuffled labels must not score. If they do, the split is leaking."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 8))
    y = rng.normal(size=300)
    assert cross_val(x, y, alpha=1.0, folds=5, purge=5)["r2"] < 0.1


def test_interleaved_blocks_still_cover_every_sample_once():
    folds = blocked_folds(100, folds=5, purge=0, blocks=20)
    covered = np.concatenate([test for _, test in folds])
    assert sorted(covered.tolist()) == list(range(100))


def test_interleaved_training_spans_the_whole_sequence():
    """The flaw this fixes: with one long span per fold, fold 0 trains only on
    the tail and must extrapolate to the head. Measured R^2 = -25 on the real
    encoder while the same features beat the mean on absolute error."""
    contiguous_train, _ = blocked_folds(500, folds=5, purge=10)[0]
    interleaved_train, _ = blocked_folds(500, folds=5, purge=10, blocks=25)[0]
    assert contiguous_train.min() > 100          # head unseen in training
    assert interleaved_train.min() < 60          # head covered
    assert interleaved_train.max() > 440         # tail covered too


def test_interleaved_purge_still_excludes_neighbours():
    train, test = blocked_folds(500, folds=5, purge=10, blocks=25)[0]
    train_set = set(train.tolist())
    for index in test:
        for offset in range(1, 11):
            assert index + offset not in train_set or index + offset in set(test.tolist())


def test_blocks_cannot_be_fewer_than_folds():
    with pytest.raises(ValueError, match="blocks must be at least folds"):
        blocked_folds(100, folds=5, blocks=3)


def test_interleaving_rescues_a_monotonic_target():
    """A perfectly learnable monotonic signal scores terribly under contiguous
    blocks and well under interleaved ones. Same data, same estimator."""
    y = np.linspace(0, 1, 400)
    x = np.stack([y, np.zeros(400)], axis=1)
    contiguous = cross_val(x, y, alpha=1.0, folds=5, purge=5)
    interleaved = cross_val(x, y, alpha=1.0, folds=5, purge=5, blocks=40)
    assert interleaved["r2"] > contiguous["r2"]
