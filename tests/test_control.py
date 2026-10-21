import numpy as np

from poke.control import TEXT_BOX_INTERIOR, WHITE_CUT, has_control, text_box_open
from poke.games import YELLOW
from poke.ram import NO_BATTLE, WILD_BATTLE
from tests.conftest import FakeEnv


class ScreenEnv(FakeEnv):
    """FakeEnv with a paintable screen, so a test can draw a text box."""

    def __init__(self, battle=NO_BATTLE, box=False):
        super().__init__(YELLOW, {YELLOW.ram.in_battle: battle})
        self.frame = np.zeros((144, 160, 3), dtype=np.uint8)
        if box:
            self.frame[TEXT_BOX_INTERIOR] = 255

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
    interior = env.frame[TEXT_BOX_INTERIOR]
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

    assert control.MEASURED["holdout_balanced_accuracy"] == 0.9494
    assert control.MEASURED["fit_balanced_accuracy"] == 0.9371
    assert "results/control_calibration.json" in control.MEASURED["report"]
