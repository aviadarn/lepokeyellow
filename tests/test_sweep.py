import numpy as np

from scripts.sweep_seeds import rank_auc


def test_perfect_separation_when_cheap_plans_succeed():
    costs = np.array([0.1, 0.2, 0.8, 0.9])
    successes = np.array([True, True, False, False])
    assert rank_auc(costs, successes) == 1.0


def test_inverted_separation():
    costs = np.array([0.1, 0.2, 0.8, 0.9])
    successes = np.array([False, False, True, True])
    assert rank_auc(costs, successes) == 0.0


def test_all_ties_is_exactly_chance():
    """The diagnosis the sweep exists to make: identical predicted costs mean
    the planner cannot tell its good plans from its bad ones."""
    costs = np.full(8, 0.5)
    successes = np.array([True, False] * 4)
    assert rank_auc(costs, successes) == 0.5


def test_one_sided_outcome_is_undefined_not_zero():
    costs = np.array([0.1, 0.2, 0.3])
    assert rank_auc(costs, np.array([True, True, True])) is None
    assert rank_auc(costs, np.array([False, False, False])) is None


def test_single_tie_pair_counts_as_half():
    costs = np.array([0.5, 0.5])
    assert rank_auc(costs, np.array([True, False])) == 0.5


def test_matches_a_brute_force_pair_count():
    rng = np.random.default_rng(0)
    for _ in range(20):
        costs = rng.integers(0, 5, size=12).astype(float)  # ties on purpose
        successes = rng.random(12) < 0.5
        if successes.all() or not successes.any():
            continue
        positive, negative = costs[successes], costs[~successes]
        wins = sum(
            1.0 if p < n else 0.5 if p == n else 0.0
            for p in positive for n in negative
        )
        expected = wins / (len(positive) * len(negative))
        assert abs(rank_auc(costs, successes) - expected) < 1e-9
