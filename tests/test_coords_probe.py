import numpy as np
import pytest
import torch as t

from poke.coords_probe import aggregate_cost, decode
from poke.probe_mlp import fit_mlp, stack


def linear_probe(dim=4, maps=2):
    """A probe dict with hand-set weights: map m reads y = x[m], x = x[m] * 2."""
    return {
        "map_w": t.eye(dim)[:, :maps],
        "map_b": t.zeros(maps),
        "y_w": t.eye(dim)[:maps],
        "y_b": t.zeros(maps),
        "x_w": t.eye(dim)[:maps] * 2.0,
        "x_b": t.zeros(maps),
        "maps": t.arange(maps),
    }


def test_decode_keeps_the_leading_shape():
    probe = linear_probe()
    states = t.randn(5, 7, 4)
    map_index, y, x = decode(states, probe)
    assert map_index.shape == y.shape == x.shape == (5, 7)


def test_decode_reads_the_head_of_the_map_it_picked():
    probe = linear_probe()
    states = t.tensor([[[3.0, 1.0, 0.0, 0.0]], [[1.0, 4.0, 0.0, 0.0]]])
    map_index, y, x = decode(states, probe)
    assert map_index.flatten().tolist() == [0, 1]
    assert y.flatten().tolist() == [3.0, 4.0]
    assert x.flatten().tolist() == [6.0, 8.0]


def test_decode_can_be_told_which_map_it_is():
    """A measurement wants positional error on its own, not mixed with map
    classification error."""
    probe = linear_probe()
    states = t.tensor([[[3.0, 1.0, 0.0, 0.0]]])
    forced = t.ones(1, 1, dtype=t.long)
    map_index, y, x = decode(states, probe, map_index=forced)
    assert map_index.flatten().tolist() == [1]
    assert y.flatten().tolist() == [1.0]


def test_chunking_does_not_change_the_answer():
    probe = linear_probe()
    states = t.randn(9, 5, 4)
    whole = decode(states, probe)
    chunked = decode(states, probe, chunk=3)
    for a, b in zip(whole, chunked):
        assert t.equal(a, b)


def test_decode_runs_the_mlp_heads_when_they_are_present():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(200, 4))
    heads = [fit_mlp(features, features[:, index], epochs=30, seed=0)
             for index in range(2)]
    probe = linear_probe()
    probe["y_mlp"] = stack(heads, t.device("cpu"))
    probe["x_mlp"] = stack(heads, t.device("cpu"))

    states = t.tensor([[[3.0, 1.0, 0.0, 0.0]]])
    _, y, x = decode(states, probe)
    assert y.shape == (1, 1) and t.equal(y, x)


def test_min_aggregation_takes_the_best_state_anywhere():
    trace = t.tensor([[5.0, 1.0, 9.0], [4.0, 4.0, 4.0]])
    assert aggregate_cost(trace, "min").tolist() == [1.0, 4.0]


def test_final_aggregation_scores_only_where_the_plan_ends():
    trace = t.tensor([[5.0, 1.0, 9.0]])
    assert aggregate_cost(trace, "final").tolist() == [9.0]


def test_discounted_aggregation_weights_the_imminent_steps_most():
    """`min` over a long lookahead scores a plan by the single most optimistic
    tile the head decoded anywhere along it, which with a read-out whose error
    grows with the rollout is a search for the head's largest mistake."""
    early = t.tensor([[0.0, 10.0, 10.0]])
    late = t.tensor([[10.0, 10.0, 0.0]])
    assert (aggregate_cost(early, "discounted", 0.5)
            < aggregate_cost(late, "discounted", 0.5))
    # min cannot tell them apart at all.
    assert aggregate_cost(early, "min") == aggregate_cost(late, "min")


def test_discounted_aggregation_is_an_average_not_a_sum():
    """So the number stays in hops and stays comparable across horizons."""
    flat = t.full((1, 6), 3.0)
    assert abs(float(aggregate_cost(flat, "discounted", 0.9)) - 3.0) < 1e-5


def test_unknown_aggregation_is_an_error():
    with pytest.raises(ValueError, match="unknown cost aggregation"):
        aggregate_cost(t.zeros(1, 2), "best")
