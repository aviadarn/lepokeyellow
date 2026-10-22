"""Find the WRAM byte that tracks "the player has control".

`scripts/play.py` answers that question today by snapshotting, pressing two
different directions and comparing screens. It is correct, and at play time it
is banned, because the contract forbids rollback. So: run the game, take the
expensive answer as ground truth, and look for a byte that agrees with it.

Scoring is the best deterministic rule a single byte supports -- the set of
values on which the player has control -- rather than the narrower "zero means
control". A first pass with the narrow rule topped out at 0.797 agreement,
which is not a probe, it is a coin with a bias.

Overfitting is the obvious risk in fitting a value set to 300 samples, so the
fit and the check come from different save states: --state is given twice or
more, the last one is held out, and the report carries both numbers. Same
shape as scripts/calibrate_ram.py otherwise: scan, score, print, pin by hand.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from poke.actions import ACTIONS
from poke.env import PokeEnv

WRAM_START, WRAM_STOP = 0xC000, 0xE000
# A rule over more values than this is fitting noise, not reading a flag.
MAX_VALUE_SET = 4


def ground_truth(env) -> bool:
    """The rollback answer: two buttons from one state, two different screens."""
    blob = env.snapshot()
    screens = []
    try:
        for button in ("down", "right"):
            env.restore(blob)
            env.step(ACTIONS.index(button))
            screens.append(env.screen()[:, :, 0].tobytes())
    finally:
        env.restore(blob)
    return screens[0] != screens[1]


def screen_features(screen):
    """A handful of scalars a text box moves and the overworld does not.

    Gen-1 draws its box over the bottom six tile rows: a frame, then a white
    interior with black glyphs. The overworld down there is mid-grey tiles.
    """
    bottom = screen[96:144, :, 0]
    interior = screen[104:140, 8:152, 0]
    return {
        "white_bottom": float((bottom == 255).mean()),
        "white_interior": float((interior == 255).mean()),
        "black_bottom": float((bottom == 0).mean()),
        "unique_bottom": float(len(set(bottom.reshape(-1).tolist()))),
        "mean_border_row": float(screen[100, :, 0].mean()),
    }


def collect(rom, state, samples, seed):
    """[(has_control, wram bytes, screen features)] over a random walk.

    Battle turns are dropped. In a battle the menu is on screen and the d-pad
    moves the cursor, so the rollback probe says "control" while a box is
    plainly drawn -- which teaches a box detector the opposite of the truth.
    poke.control handles battles with in_battle, not with this probe.
    """
    rng = random.Random(seed)
    env = PokeEnv(rom, state)
    env.reset()
    env.settle()
    rows = []
    for _ in range(samples):
        if not env.read_ram(env.game.ram.in_battle):
            rows.append((ground_truth(env),
                         env.read_ram_range(WRAM_START, WRAM_STOP),
                         screen_features(env.screen())))
        env.step(rng.randrange(len(ACTIONS)))
    env.close()
    return rows


def tally(rows):
    """address -> value -> [no-control count, control count]."""
    counts = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for truth, wram, _ in rows:
        for offset, value in enumerate(wram):
            counts[WRAM_START + offset][value][int(truth)] += 1
    return counts


def best_threshold(rows, name):
    """The best 'control iff feature <= t' (or '>= t') rule for one feature.

    Among cuts that score the same, the widest-margin one wins. This is not a
    nicety. The first version took the midpoint of the first two adjacent
    observed values that tied for best, which put the cut *inside* the
    overworld's own spread: on Pallet Town's sand the box detector then read a
    box on walkable ground, and because only the policy can walk off a tile,
    that reading was absorbing. The margin is what keeps the cut in the gap.
    """
    values = sorted({row[2][name] for row in rows})
    cuts = [(a + b) / 2 for a, b in zip(values, values[1:])] or values
    scored = []
    for cut in cuts:
        for sense in ("<=", ">"):
            hits = [0, 0]
            totals = [0, 0]
            for truth, _, features in rows:
                predicted = (features[name] <= cut) if sense == "<=" else (features[name] > cut)
                totals[int(truth)] += 1
                hits[int(truth)] += int(predicted == truth)
            recalls = [hits[i] / totals[i] for i in (0, 1) if totals[i]]
            score = sum(recalls) / len(recalls) if recalls else 0.0
            scored.append((round(score, 6), cut, sense))
    if not scored:
        return 0.0, None, None
    best_score = max(row[0] for row in scored)
    tied = [row for row in scored if row[0] == best_score]

    def margin(cut):
        """Distance to the nearest sample the cut is deciding between."""
        below = [v for v in values if v <= cut]
        above = [v for v in values if v > cut]
        if not below or not above:
            return 0.0
        return min(cut - max(below), min(above) - cut)

    return max(tied, key=lambda row: margin(row[1]))


def score_threshold(rows, name, cut, sense):
    hits = [0, 0]
    totals = [0, 0]
    for truth, _, features in rows:
        predicted = (features[name] <= cut) if sense == "<=" else (features[name] > cut)
        totals[int(truth)] += 1
        hits[int(truth)] += int(predicted == truth)
    recalls = [hits[i] / totals[i] for i in (0, 1) if totals[i]]
    return sum(recalls) / len(recalls) if recalls else 0.0


def best_rule(per_value):
    """The best 'control iff value in S' rule for one byte, and its score.

    The optimal S assigns each observed value to whichever class it mostly
    carries. Scoring is *balanced* accuracy -- the mean of the two per-class
    recalls -- because the classes are nowhere near even: on a state where the
    player can move 92% of the time, "always yes" scores 0.92 and reads like a
    probe.
    """
    control_values = sorted(v for v, pair in per_value.items() if pair[1] > pair[0])
    return balanced_accuracy(per_value, control_values), control_values


def balanced_accuracy(per_value, control_values):
    """How a rule scores on a tally: mean of the two per-class recalls."""
    allowed = set(control_values)
    hits = [0, 0]
    totals = [0, 0]
    for value, (without, with_) in per_value.items():
        totals[0] += without
        totals[1] += with_
        if value in allowed:
            hits[1] += with_
        else:
            hits[0] += without
    recalls = [hits[i] / totals[i] for i in (0, 1) if totals[i]]
    return sum(recalls) / len(recalls) if recalls else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rom", default="pokeyellow.gbc")
    parser.add_argument("--state", nargs="+",
                        default=["states/pallet.state", "states/pikachu.state",
                                 "states/oaks_lab.state", "states/bedroom.state",
                                 "states/oracle/route1.state",
                                 "states/oracle/forest_gate.state"],
                        help="fit on all but the last; the last is held out")
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/control_calibration.json")
    args = parser.parse_args()

    if len(args.state) < 2:
        parser.error("give at least two --state files: one to fit, one to hold out")

    fit_rows = []
    for index, state in enumerate(args.state[:-1]):
        fit_rows += collect(args.rom, state, args.samples, args.seed + index)
    holdout_rows = collect(args.rom, args.state[-1], args.samples,
                           args.seed + len(args.state))

    fit, holdout = tally(fit_rows), tally(holdout_rows)
    # Every address is scored on the held-out state, not just the ones that
    # looked good on the fit: a byte of the tile buffer can memorise the fit
    # set perfectly and score 0.08 on a state it never saw, and truncating by
    # fit score first is how such a byte ends up being the winner.
    scored = []
    for addr, per_value in fit.items():
        fit_score, control_values = best_rule(per_value)
        if len(control_values) > MAX_VALUE_SET:
            continue
        scored.append({
            "address": addr,
            "address_hex": f"{addr:#06x}",
            "fit_accuracy": round(fit_score, 4),
            "holdout_accuracy": round(balanced_accuracy(holdout[addr], control_values), 4),
            "control_values": control_values,
        })
    scored.sort(key=lambda row: (-row["holdout_accuracy"], -row["fit_accuracy"]))
    ranked = scored[:10]
    best = ranked[0]

    feature_rows = []
    for name in fit_rows[0][2]:
        fit_score, cut, sense = best_threshold(fit_rows, name)
        feature_rows.append({
            "feature": name, "rule": f"control iff {name} {sense} {cut:.6g}",
            "fit_accuracy": round(fit_score, 4),
            "holdout_accuracy": round(score_threshold(holdout_rows, name, cut, sense), 4),
            "cut": cut, "sense": sense,
        })
    feature_rows.sort(key=lambda row: (-row["holdout_accuracy"], -row["fit_accuracy"]))
    # The winning feature's own samples, per class, so a unit test can pin the
    # cut inside the gap without a ROM. Rounded and deduplicated: this is a
    # fixture, not a dataset.
    won = feature_rows[0]["feature"]
    samples = {"with_control": set(), "without_control": set()}
    for truth, _, features in fit_rows + holdout_rows:
        key = "with_control" if truth else "without_control"
        samples[key].add(round(features[won], 6))
    feature_samples = {
        "feature": won,
        "cut": feature_rows[0]["cut"],
        "sense": feature_rows[0]["sense"],
        "with_control": sorted(samples["with_control"]),
        "without_control": sorted(samples["without_control"]),
    }

    report = {
        "winner": best,
        "screen_features": feature_rows,
        "feature_samples": feature_samples,
        "fit_states": args.state[:-1],
        "holdout_state": args.state[-1],
        "samples_per_state": args.samples,
        "fit_with_control": sum(1 for truth, _, _ in fit_rows if truth),
        "fit_without_control": sum(1 for truth, _, _ in fit_rows if not truth),
        "holdout_with_control": sum(1 for truth, _, _ in holdout_rows if truth),
        "holdout_without_control": sum(1 for truth, _, _ in holdout_rows if not truth),
        "no_control_values": dict(Counter(
            value for truth, wram, _ in fit_rows + holdout_rows
            if not truth
            for value in (wram[best["address"] - WRAM_START],)
        ).most_common()),
        "runners_up": ranked[1:],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "runners_up"}, indent=1))
    print("best RAM byte:", json.dumps(best))
    print("runners up:", json.dumps(report["runners_up"]))
    thin = min(report["fit_with_control"], report["fit_without_control"],
               report["holdout_with_control"], report["holdout_without_control"])
    if thin < 20:
        print(f"WARNING: thinnest class has {thin} samples; the winner is not "
              "evidence. Re-run from states where text boxes happen.")
    if best["holdout_accuracy"] < 0.99:
        print("Accuracy below is BALANCED accuracy: the mean of the two "
              "per-class recalls, not the raw hit rate.")
        print("WARNING: held-out accuracy is under 99%. Do not pin this address.")


if __name__ == "__main__":
    main()
