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
The box's white bottom rows are a real signal and that is what is pinned here.

It is not a clean separation. The two classes overlap by one sample, so a
walkable tile can read as a box; `poke.handlers.BOX_PATIENCE` is what keeps
that from being absorbing.

What it cannot see, by design: battles. In a battle a box is on screen and the
d-pad works the menu, so `in_battle` separates them, not this detector.
"""

from __future__ import annotations

from poke.ram import in_battle

#: The bottom six tile rows, where Gen-1 draws its box. Full width: the frame
#: counts, and cropping to the interior scored no better.
TEXT_BOX_REGION = (slice(96, 144), slice(0, 160), 0)

#: Control iff that region is less white than this. Fitted, not chosen, and
#: among cuts that tie on accuracy it is the widest-margin one -- the first
#: version took the midpoint of the first tied pair, which landed inside the
#: overworld's own spread and read a box on Pallet Town's sand.
WHITE_CUT = 0.7724609375

#: Where that number came from, and how well it did. A tuned constant without
#: its accuracy beside it is a magic number, and this one is quoted in the
#: README wherever the learned share is: the share carries the same error.
MEASURED = {
    "report": "results/control_calibration.json",
    "feature": "white fraction of rows 96:144",
    "fit_states": 5,
    "fit_samples": 1046,
    "fit_balanced_accuracy": 0.9503,
    "holdout_state": "states/pikachu.state",
    "holdout_samples": 250,
    "holdout_balanced_accuracy": 0.9835,
    "rejected": "0xdff5 (0.966 fit / 1.000 holdout) -- stack residue, not a flag",
    # The two classes overlap: one walkable sample reads as white as a box.
    # No cut removes it, so poke.handlers.BOX_PATIENCE bounds what a wrong
    # reading can cost instead of pretending the reading is always right.
    "overlapping_samples": 1,
}


def text_box_open(env) -> bool:
    """True while a text box or menu is drawn over the bottom of the screen.

    Glyphs are black, so a box is never fully white; the cut is a fraction.
    """
    region = env.screen()[TEXT_BOX_REGION]
    return float((region == 255).mean()) > WHITE_CUT


def has_control(env) -> bool:
    """True when an overworld button from the policy would do something: no
    box drawn, no battle running."""
    return not text_box_open(env) and not in_battle(env)
