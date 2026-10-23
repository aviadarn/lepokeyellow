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

import numpy as np

from poke.ram import in_battle

#: The bottom six tile rows, where Gen-1 draws its box. Full width: the frame
#: counts, and cropping to the interior scored no better.
TEXT_BOX_REGION = (slice(96, 144), slice(0, 160), 0)

#: How many distinct greys a text box may use. A Gen-1 box is white paper
#: plus black frame and glyphs; the overworld uses all four DMG shades. This
#: is the discriminating feature, measured at 0.9683 balanced accuracy where
#: the white fraction alone reached 0.9497.
BOX_SHADES = 2

#: And the paper has to be there, so a fade to black -- two shades, no white
#: -- is not mistaken for a box. Well clear of both classes' mass.
WHITE_FLOOR = 0.4

#: The fraction the detector used to be, kept because the calibration report
#: and the README both quote it, and because the shade rule is judged against
#: it. On its own it read a box on Pallet Town's entrance tiles and trapped
#: the agent there; see MEASURED["superseded"].
WHITE_CUT = 0.7724609375

#: Where these numbers came from, and how well the rule did. A tuned constant
#: without its accuracy beside it is a magic number, and this one is quoted in
#: the README wherever the learned share is: the share carries the same error.
MEASURED = {
    "report": "results/control_calibration_v2.json",
    "feature": "distinct shades in rows 96:144, with a white floor",
    "fit_states": 6,
    "fit_samples": 1314,
    "fit_balanced_accuracy": 0.9683,
    # The number that actually matters. A walkable tile read as a box is
    # absorbing -- only the policy can step off a tile, and the handler owns
    # every turn it claims -- so this is the count of tiles the agent can be
    # trapped on, not a percentage point.
    "walkable_called_a_box": 0,
    "walkable_samples": 967,
    # There is no honest holdout number. The ground truth is the rollback
    # probe: press two buttons, compare screens. In a menu the d-pad moves a
    # cursor, so the probe reports control while a box is plainly drawn, and
    # 87.5% of the holdout state's "walkable" samples are menus. Both rules
    # score 0.5106 there, which measures the probe and not the detector.
    "holdout_state": "states/pikachu.state",
    "holdout_is_menus": 0.875,
    "holdout_balanced_accuracy": None,
    "superseded": {
        "report": "results/control_calibration.json",
        "feature": "white fraction of rows 96:144",
        "cut": WHITE_CUT,
        # On the same six-state sample set the new rule is scored against,
        # so the two numbers are comparable. On its own five-state set it
        # scored 0.9503 fit / 0.9835 holdout, which is what the README quoted
        # while the agent was being trapped -- the samples never visited the
        # tile that did it.
        "fit_balanced_accuracy": 0.9497,
        "fit_balanced_accuracy_on_its_own_samples": 0.9503,
        # What the percentage point hid: Pallet Town's entrance is 82% white,
        # so 36 of 967 walkable samples read as a text box. One of them is
        # (0, 4, 11), where a trained policy spent 1,411 of its 1,461 actions
        # with the handler taking 32 turns in every 33.
        "walkable_called_a_box": 36,
    },
    "rejected": "0xdff5 (0.966 fit / 1.000 holdout) -- stack residue, not a flag",
}


def text_box_open(env) -> bool:
    """True while a text box or menu is drawn over the bottom of the screen.

    Two conditions, and the shade count is the one that matters. Gen-1 draws
    its box in two colours -- white paper and black frame-and-glyphs -- while
    scenery uses all four shades the DMG has. Whiteness alone cannot separate
    them: Pallet Town's ground is 82% white, whiter than plenty of real boxes.

    The white floor is what keeps a fade (two shades, no paper) from reading
    as a box.
    """
    region = env.screen()[TEXT_BOX_REGION]
    if len(np.unique(region)) > BOX_SHADES:
        return False
    return float((region == 255).mean()) > WHITE_FLOOR


def has_control(env) -> bool:
    """True when an overworld button from the policy would do something: no
    box drawn, no battle running."""
    return not text_box_open(env) and not in_battle(env)
