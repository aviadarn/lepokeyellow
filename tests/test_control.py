import numpy as np

from poke.control import TEXT_BOX_REGION, WHITE_CUT, has_control, text_box_open
from poke.games import YELLOW
from poke.ram import NO_BATTLE, WILD_BATTLE
from tests.conftest import FakeEnv


class ScreenEnv(FakeEnv):
    """FakeEnv with a paintable screen, so a test can draw a text box."""

    def __init__(self, battle=NO_BATTLE, box=False):
        super().__init__(YELLOW, {YELLOW.ram.in_battle: battle})
        self.frame = np.zeros((144, 160, 3), dtype=np.uint8)
        if box:
            self.frame[TEXT_BOX_REGION] = 255

    def screen(self):
        return self.frame


def test_control_is_free_when_no_box_is_drawn():
    assert has_control(ScreenEnv())
    assert not text_box_open(ScreenEnv())


def test_a_white_box_over_the_bottom_rows_takes_control_away():
    assert text_box_open(ScreenEnv(box=True))
    assert not has_control(ScreenEnv(box=True))


def test_the_cut_is_a_fraction_not_every_pixel():
    """Glyphs are black, so a real box is never 100% white. The measured cut
    sits at 0.8164, and a box that is 90% white still reads as a box."""
    env = ScreenEnv(box=True)
    interior = env.frame[TEXT_BOX_REGION]
    interior.reshape(-1, 3)[: int(interior.size / 3 * 0.10)] = 0
    assert 0.0 < WHITE_CUT < 1.0
    assert text_box_open(env)


def test_a_battle_is_not_control_even_with_no_box():
    """The battle handler owns these turns, and the detector is blind to them
    on purpose: in a battle the d-pad works the menu, so a box is on screen
    while input is live. in_battle is what separates them."""
    assert not has_control(ScreenEnv(battle=WILD_BATTLE))


def test_the_probe_never_snapshots():
    """Play time bans rollback, so the probe must not be able to use it."""
    class NoRollbackEnv(ScreenEnv):
        def snapshot(self):
            raise AssertionError("has_control must not snapshot at play time")

        def restore(self, blob):
            raise AssertionError("has_control must not restore at play time")

    assert has_control(NoRollbackEnv())


def test_the_measured_accuracy_travels_with_the_threshold():
    """A tuned constant with no recorded accuracy is a magic number."""
    from poke import control

    import json
    from pathlib import Path

    report = json.loads(Path(control.MEASURED["report"]).read_text())
    winner = report["screen_features"][0]
    assert control.MEASURED["fit_balanced_accuracy"] == winner["fit_accuracy"]
    assert control.MEASURED["holdout_balanced_accuracy"] == winner["holdout_accuracy"]
    assert control.WHITE_CUT == winner["cut"]


def test_the_cut_sits_in_the_gap_between_the_recorded_classes():
    """The calibration's own samples, replayed without a ROM.

    This is the regression test for the Critical the review found: the first
    cut sat inside the overworld's spread, so Pallet Town's sand read as a
    text box, and because only the policy can walk off a tile the reading was
    absorbing. The cut must sit above the walkable mass.
    """
    import json
    from pathlib import Path

    from poke import control

    report = json.loads(Path("results/control_calibration.json").read_text())
    samples = report["feature_samples"]
    assert samples["feature"] == "white_bottom"
    assert samples["cut"] == control.WHITE_CUT

    walkable = samples["with_control"]
    absorbing = [value for value in walkable if value > control.WHITE_CUT]
    assert len(absorbing) <= control.MEASURED["overlapping_samples"], (
        f"{len(absorbing)} walkable samples read as a text box; the handler "
        "owns every turn it claims, so each one is a tile the policy can be "
        "trapped on"
    )
    below = [value for value in walkable if value <= control.WHITE_CUT]
    assert max(below) < control.WHITE_CUT
