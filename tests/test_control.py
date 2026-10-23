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


def test_the_measured_accuracy_travels_with_the_rule():
    """A tuned constant with no recorded accuracy is a magic number."""
    import json
    from pathlib import Path

    from poke import control

    report = json.loads(Path(control.MEASURED["report"]).read_text())
    assert report["constants"]["BOX_SHADES"] == control.BOX_SHADES
    assert report["constants"]["WHITE_FLOOR"] == control.WHITE_FLOOR
    assert (control.MEASURED["fit_balanced_accuracy"]
            == report["rules"]["two_shade"]["fit_balanced_accuracy"])
    assert (control.MEASURED["superseded"]["fit_balanced_accuracy"]
            == report["rules"]["white_cut"]["fit_balanced_accuracy"])


def test_no_walkable_tile_reads_as_a_text_box():
    """The real regression, and the one the old version of this test missed.

    It used to assert the cut sat above the *recorded* walkable samples, and
    it passed -- while an agent sat trapped on Pallet Town's entrance for
    1,411 of 1,461 actions. The samples simply did not include that tile.
    The check that matters is the count over a walk that does visit it, and
    for an absorbing failure the only acceptable count is zero: the handler
    owns every turn it claims, and only the policy can step off a tile.
    """
    import json
    from pathlib import Path

    from poke import control

    report = json.loads(Path(control.MEASURED["report"]).read_text())
    trapped = report["walkable_called_a_box"]["two_shade"]
    assert trapped == control.MEASURED["walkable_called_a_box"]
    assert trapped == 0, (
        f"{trapped} of {report['walkable_samples']} walkable samples read as "
        "a text box; each one is a tile the policy can be trapped on"
    )
    assert report["walkable_called_a_box"]["white_cut"] == 36, (
        "the rule this replaced, on the same samples"
    )


def test_the_holdout_split_is_recorded_as_unusable():
    """Honesty about what was not measured.

    The rollback probe cannot label a menu: the d-pad moves a cursor, so two
    buttons give two screens and it reports control with a box on screen.
    The holdout state is 87.5% menus, so both rules score chance there. That
    is a fact about the probe, and it belongs beside the number it voids.
    """
    from poke import control

    assert control.MEASURED["holdout_balanced_accuracy"] is None
    assert control.MEASURED["holdout_is_menus"] > 0.5


def _bottom(shades):
    """A screen whose bottom six rows carry exactly `shades` (value, count)."""
    frame = np.zeros((144, 160, 3), dtype=np.uint8)
    region = np.full((48, 160), 255, dtype=np.uint8)
    flat = region.reshape(-1)
    at = 0
    for value, count in shades:
        flat[at:at + count] = value
        at += count
    frame[96:144, :, 0] = region
    return frame


class PaintedEnv(FakeEnv):
    def __init__(self, frame):
        super().__init__(YELLOW, {YELLOW.ram.in_battle: NO_BATTLE})
        self.frame = frame

    def screen(self):
        return self.frame


def test_pale_scenery_in_four_shades_is_not_a_text_box():
    """Pallet Town's entrance. 82% of the bottom rows are white and there is
    no box on screen; the white-fraction cut called it a box and the handler
    then owned 32 of every 33 turns. Measured on the cartridge at (0, 4, 11),
    where the agent spent 1,411 of 1,461 actions.

    A Gen-1 box is drawn in two shades -- white paper, black glyphs. Scenery
    uses all four the DMG has.
    """
    frame = _bottom([(170, 700), (85, 500), (0, 182)])
    white = float((frame[TEXT_BOX_REGION] == 255).mean())
    assert white > WHITE_CUT, "this frame is whiter than the old cut, by construction"
    assert not text_box_open(PaintedEnv(frame))
    assert has_control(PaintedEnv(frame))


def test_white_paper_with_black_glyphs_is_a_text_box():
    frame = _bottom([(0, 900)])
    assert text_box_open(PaintedEnv(frame))


def test_a_dark_fade_is_not_a_text_box():
    """Two shades, but no paper: the screen is fading, not talking."""
    frame = _bottom([(0, 7000)])
    assert not text_box_open(PaintedEnv(frame))
