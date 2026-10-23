"""The rules the box detector is chosen between, scored without a ROM."""

import pytest

from scripts.calibrate_box import RULES, balanced_accuracy, features_of


def test_the_two_shade_rule_rejects_four_shade_scenery():
    """The Pallet Town entrance, as the cartridge measures it."""
    pallet = {"white_bottom": 0.818, "unique_bottom": 4}
    assert RULES["white_cut"](pallet) is True, "the old rule called this a box"
    assert RULES["two_shade"](pallet) is False


def test_the_two_shade_rule_accepts_a_real_box():
    box = {"white_bottom": 0.808, "unique_bottom": 2}
    assert RULES["two_shade"](box) is True


def test_a_fade_has_two_shades_and_is_still_not_a_box():
    assert RULES["two_shade"]({"white_bottom": 0.0, "unique_bottom": 1}) is False


def test_balanced_accuracy_averages_the_two_classes():
    """Not raw accuracy: the classes are far from even, and a rule that calls
    everything 'control' would score well on raw."""
    rows = [(False, {"white_bottom": 1.0, "unique_bottom": 2})] * 2 + \
           [(True, {"white_bottom": 0.0, "unique_bottom": 4})] * 98
    assert balanced_accuracy(rows, RULES["two_shade"]) == pytest.approx(1.0)
    everything_is_control = lambda f: False
    assert balanced_accuracy(rows, everything_is_control) == pytest.approx(0.5)


def test_features_of_counts_shades_and_white():
    import numpy as np
    frame = np.zeros((144, 160, 3), dtype=np.uint8)
    frame[96:144, :, 0] = 255
    frame[96:100, :, 0] = 0
    f = features_of(frame)
    assert f["unique_bottom"] == 2
    assert f["white_bottom"] == pytest.approx(1 - 4 / 48)
