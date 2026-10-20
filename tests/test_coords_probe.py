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


@pytest.fixture
def dataset(tmp_path):
    """A tiny recorded walk: two episodes across two maps, with positions."""
    from poke.writer import EpisodeWriter

    rom = tmp_path / "f.gb"; rom.write_bytes(b"\x00" * 1024)
    state = tmp_path / "s.state"; state.write_bytes(b"\x01" * 16)
    route = tmp_path / "r.json"; route.write_text("{}")
    out = tmp_path / "poke.h5"

    rng = np.random.default_rng(0)
    with EpisodeWriter(out, rom, state, 60, 0, game="yellow",
                       task="yellow_pikachu", route=route) as writer:
        for episode in range(2):
            for step in range(120):
                map_id = 0 if step < 60 else 37
                writer.add_step(
                    rng.integers(0, 255, (144, 160), dtype=np.uint8),
                    np.eye(6, dtype=np.uint8)[step % 6],
                    0, False, (map_id, step % 8, (step * 3) % 8),
                )
            writer.end_episode("random", f"r{episode}", False, -1, -1)
    return str(out)


@pytest.fixture
def model():
    from poke.jepa import JEPA

    t.manual_seed(0)
    jepa = JEPA(num_frames=4)
    jepa.eval()
    return jepa


def test_fit_reads_every_map_in_the_data(dataset, model):
    from poke import coords_probe

    probe = coords_probe.fit(model, dataset, t.device("cpu"), head="linear",
                             min_frames=16)
    assert probe.maps == [0, 37]
    assert probe.y_w.shape == (2, model.dim)
    assert probe.frames_per_map == {0: 120, 37: 120}
    assert probe.fitted_on == "real"


def test_fit_can_build_features_from_predicted_embeddings(dataset, model):
    """Off by default and kept because the result was instructive: see the
    docstring in poke/coords_probe.py."""
    from poke import coords_probe

    probe = coords_probe.fit(model, dataset, t.device("cpu"), head="linear",
                             min_frames=16, on="predicted", steps=2)
    assert probe.fitted_on == "predicted"
    # Two rollout lengths per usable frame, minus the ones that run off the end
    # of their episode.
    assert sum(probe.frames_per_map.values()) > 240


def test_fit_restricts_itself_to_the_episodes_it_is_given(dataset, model):
    from poke import coords_probe

    probe = coords_probe.fit(model, dataset, t.device("cpu"), head="linear",
                             min_frames=16, only_episodes=[0])
    assert sum(probe.frames_per_map.values()) == 120


def test_the_mlp_path_reports_held_out_error_per_map(dataset, model):
    from poke import coords_probe

    probe = coords_probe.fit(model, dataset, t.device("cpu"), head="mlp",
                             min_frames=16, epochs=20, patience=5)
    assert set(probe.holdout_tile_mae) == {"map0_y", "map0_x",
                                           "map37_y", "map37_x"}
    assert len(probe.y_heads) == 2 and len(probe.x_heads) == 2
    assert "y_mlp" in probe.to_torch(t.device("cpu"))


def test_an_unknown_head_or_feature_source_is_an_error(dataset, model):
    from poke import coords_probe

    with pytest.raises(ValueError, match="head must be"):
        coords_probe.fit(model, dataset, t.device("cpu"), head="svm")
    with pytest.raises(ValueError, match="on must be"):
        coords_probe.fit(model, dataset, t.device("cpu"), on="future")
