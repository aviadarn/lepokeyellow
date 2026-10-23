"""Choose the text-box detector by measurement, not by taste.

The first detector was the white fraction of the screen's bottom six rows.
It scored 0.9503 and shipped, and then an agent spent 1,411 of its 1,461
actions standing on one tile in Pallet Town: the ground there is 82% white,
whiter than the cut, so the handler claimed every turn it was allowed to.
`poke.handlers.BOX_PATIENCE` bounded that at one policy turn in thirty-three,
which is not control.

Whiteness was never the signal. A Gen-1 text box is drawn in two shades --
white paper, black frame and glyphs -- and scenery uses all four the DMG has.
This script scores both rules against the rollback probe in
`scripts/calibrate_control.ground_truth`, fitting on states that include the
tiles the old rule failed on, and writes the report the constants are pinned
to.

Ground truth is itself approximate, and the report says where: in a menu the
d-pad moves a cursor, so two different buttons give two different screens and
the probe reports control while a box is plainly drawn. Battles are dropped
for the same reason. A state whose samples are mostly menus scores both rules
identically and measures nothing.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from poke.actions import ACTIONS
from poke.control import BOX_SHADES, WHITE_CUT, WHITE_FLOOR
from poke.env import PokeEnv
from poke.ram import in_battle

#: The bottom six tile rows, as a plain slice so this script does not depend
#: on the constant it is scoring.
BOTTOM = (slice(96, 144), slice(0, 160), 0)

FIT_STATES = [
    "states/bedroom.state",
    "states/oaks_lab.state",
    "states/pallet.state",
    "states/oracle/route1.state",
    "states/oracle/parcel.state",
    "states/oracle/forest_gate.state",
]
HOLDOUT_STATE = "states/pikachu.state"

RULES = {
    "white_cut": lambda f: bool(f["white_bottom"] > WHITE_CUT),
    "two_shade": lambda f: bool(f["unique_bottom"] <= BOX_SHADES
                                and f["white_bottom"] > WHITE_FLOOR),
}


def features_of(screen) -> dict:
    """The two numbers the rules are built from."""
    region = screen[BOTTOM]
    return {
        "white_bottom": float((region == 255).mean()),
        "unique_bottom": int(len(np.unique(region))),
    }


def balanced_accuracy(rows, rule) -> float | None:
    """Mean of the two per-class rates.

    `rows` is [(has_control, features)]. Raw accuracy rewards a rule that
    always answers with the majority class; these classes are nowhere near
    even.
    """
    boxes = [f for control, f in rows if not control]
    free = [f for control, f in rows if control]
    if not boxes or not free:
        return None
    caught = sum(1 for f in boxes if rule(f)) / len(boxes)
    left_alone = sum(1 for f in free if not rule(f)) / len(free)
    return (caught + left_alone) / 2


def walk(state, samples, seed, rom="pokeyellow.gbc"):
    """A random walk that presses A, so real boxes appear as well as scenery."""
    from scripts.calibrate_control import ground_truth

    directions = [ACTIONS.index(d) for d in ("up", "down", "left", "right")]
    a = ACTIONS.index("a")
    env = PokeEnv(rom, state)
    env.reset()
    rng = random.Random(seed)
    rows = []
    try:
        for _ in range(samples):
            env.step(rng.choice(directions * 3 + [a] * 2))
            if in_battle(env):
                for _ in range(40):
                    env.step(a)
                continue
            rows.append((ground_truth(env), features_of(env.screen())))
    finally:
        env.close()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rom", default="pokeyellow.gbc")
    parser.add_argument("--samples", type=int, default=220)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/control_calibration_v2.json")
    args = parser.parse_args()

    fit, holdout = [], []
    for index, state in enumerate(FIT_STATES):
        fit += walk(state, args.samples, args.seed + index, args.rom)
    holdout += walk(HOLDOUT_STATE, args.samples, args.seed + 100, args.rom)

    report = {
        "what": "Text-box detector: the shipped white-fraction cut against the "
                "two-shade rule, on the rollback probe.",
        "fit_states": FIT_STATES,
        "holdout_state": HOLDOUT_STATE,
        "samples": {
            "fit": len(fit),
            "fit_boxes": sum(1 for control, _ in fit if not control),
            "fit_walkable": sum(1 for control, _ in fit if control),
            "holdout": len(holdout),
            "holdout_boxes": sum(1 for control, _ in holdout if not control),
            "holdout_walkable": sum(1 for control, _ in holdout if control),
        },
        "constants": {"BOX_SHADES": BOX_SHADES, "WHITE_FLOOR": WHITE_FLOOR,
                      "WHITE_CUT": WHITE_CUT},
        "rules": {
            name: {
                "fit_balanced_accuracy": round(balanced_accuracy(fit, rule), 4),
                "holdout_balanced_accuracy": (
                    None if balanced_accuracy(holdout, rule) is None
                    else round(balanced_accuracy(holdout, rule), 4)),
            }
            for name, rule in RULES.items()
        },
        "caveat": "Ground truth is the rollback probe: two buttons, two screens. "
                  "In a menu the d-pad moves a cursor, so it reports control while "
                  "a box is drawn. Those samples count against both rules equally.",
        "walkable_called_a_box": {
            name: sum(1 for control, f in fit if control and rule(f))
            for name, rule in RULES.items()
        },
        "walkable_samples": sum(1 for control, _ in fit if control),
        # How much of each split the ground truth cannot label. A sample the
        # probe calls "control" that is two shades of mostly-white paper is a
        # menu: the d-pad moved a cursor, so two buttons gave two screens. A
        # split made mostly of these scores both rules at chance and measures
        # the probe, not the detector.
        "unlabelable_menu_fraction": {
            split: round(sum(1 for control, f in rows if control
                             and f["unique_bottom"] <= BOX_SHADES
                             and f["white_bottom"] > WHITE_FLOOR)
                         / max(sum(1 for control, _ in rows if control), 1), 4)
            for split, rows in (("fit", fit), ("holdout", holdout))
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps({"rules": report["rules"],
                      "walkable_called_a_box": report["walkable_called_a_box"],
                      "walkable_samples": report["walkable_samples"]}, indent=1))


if __name__ == "__main__":
    main()
