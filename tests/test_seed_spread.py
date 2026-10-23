"""Pooling eval runs across training seeds, and the test that compares them."""

import json

import pytest

from scripts.seed_spread import fisher_two_sided, pool, table


def test_the_tea_tasting_table_gives_fishers_own_answer():
    """Fisher's lady-tasting-tea: 3 of 4 against 1 of 4 is p = 0.4857."""
    assert fisher_two_sided(3, 4, 1, 4) == pytest.approx(0.4857, abs=5e-5)


def test_a_clean_separation_is_significant():
    """Nothing in 33 against ten in 33 -- the shape of the clone/DAgger
    comparison -- lands well under a percent."""
    assert fisher_two_sided(0, 33, 10, 33) < 0.01


def test_identical_groups_cannot_be_distinguished():
    assert fisher_two_sided(4, 10, 4, 10) == pytest.approx(1.0)


def test_pooling_counts_the_greedy_run_as_a_run(tmp_path):
    """Eval reports split the greedy run out of `reached`; it is an eleventh
    run under the contract, not a footnote, and the pooled count says so."""
    report = {"seeds": 10,
              "reached": {"get_parcel": 4, "enter_forest": 0},
              "greedy": {"objectives": {"get_parcel": 99, "enter_forest": None}}}
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(report))
    assert pool([path], "get_parcel") == (5, 11)
    assert pool([path], "enter_forest") == (0, 11)


def test_the_table_reports_each_seed_and_the_pool(tmp_path):
    paths = []
    for index, (sampled, greedy) in enumerate([(4, 99), (0, None)]):
        path = tmp_path / f"s{index}.json"
        path.write_text(json.dumps({
            "seeds": 10,
            "reached": {"get_parcel": sampled},
            "greedy": {"objectives": {"get_parcel": greedy}}}))
        paths.append(path)
    row = table({"demo": paths}, "get_parcel")["demo"]
    assert row["per_seed"] == [5, 0]
    assert row["pooled"] == [5, 22]
