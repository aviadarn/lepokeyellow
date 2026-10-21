"""Does the player have control, without rolling the emulator back.

`scripts/play.py` answers this by snapshotting, pressing two directions and
comparing screens. That is exact, and at play time it is banned: the contract
is one continuous run with no save states.

The replacement is a screen detector, chosen by measurement rather than taste.
`scripts/calibrate_control.py` scored every WRAM byte and five screen features
against the snapshot probe, fitting on three save states and checking on a
fourth it never saw. No single WRAM byte works: the best honest score was
0.962, and every top candidate sat in 0xDFEE-0xDFFB, which is the stack -- it
was reading the return address of whatever routine happened to be running.
The box's white interior is a real signal and it is the one pinned here.

What it cannot see, by design: battles. In a battle a box is on screen and the
d-pad works the menu, so `in_battle` separates them, not this detector.
"""

from __future__ import annotations

from poke.ram import in_battle

#: Rows and columns of the text box's interior, inside its frame.
TEXT_BOX_INTERIOR = (slice(104, 140), slice(8, 152), 0)

#: Control iff the interior is less white than this. Fitted, not chosen.
WHITE_CUT = 0.816358

#: Where that number came from, and how well it did. A tuned constant without
#: its accuracy beside it is a magic number, and this one is quoted in the
#: README wherever the learned share is: the share carries the same error.
MEASURED = {
    "report": "results/control_calibration.json",
    "fit_states": 3,
    "fit_samples": 612,
    "fit_balanced_accuracy": 0.9371,
    "holdout_state": "states/pallet.state",
    "holdout_samples": 300,
    "holdout_balanced_accuracy": 0.9494,
    "rejected": "0xdff5 (0.962 fit / 1.000 holdout) -- stack residue, not a flag",
}


def text_box_open(env) -> bool:
    """True while a text box or menu is drawn over the bottom of the screen.

    Glyphs are black, so a box is never fully white; the cut is a fraction.
    """
    interior = env.screen()[TEXT_BOX_INTERIOR]
    return float((interior == 255).mean()) > WHITE_CUT


def has_control(env) -> bool:
    """True when an overworld button from the policy would do something: no
    box drawn, no battle running."""
    return not text_box_open(env) and not in_battle(env)
