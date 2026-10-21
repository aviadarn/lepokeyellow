import pytest

from scripts.bench_workers import summarise


def test_perfect_scaling_is_efficiency_one():
    out = summarise([10.0, 10.0, 10.0, 10.0], single=10.0)
    assert out["workers"] == 4
    assert out["total_steps_per_s"] == 40.0
    assert out["efficiency"] == 1.0


def test_contention_shows_up_as_efficiency_below_one():
    out = summarise([5.0, 5.0], single=10.0)
    assert out["total_steps_per_s"] == 10.0
    assert out["efficiency"] == 0.5


def test_no_workers_is_an_error_not_a_zero_division():
    with pytest.raises(ValueError, match="at least one worker"):
        summarise([], single=10.0)
