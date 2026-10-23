"""How much of a Part 2 eval number is the policy, and how much is the seed?

`scripts/eval_part2.py` scores one checkpoint over ten sampling seeds and a
greedy run. That spread is the policy's own stochasticity; it says nothing
about the other spread, which turned out to be the larger one: retrain the
same dataset with a different TRAINING seed and the same pipeline scores
anywhere from zero to five. A single eval of a single checkpoint is therefore
not a measurement of the dataset that produced it, and the first DAgger result
reported from this repo -- five of eleven on `get_parcel` -- was the best of
three draws reported as if it were the number.

This script pools several training seeds per condition and runs Fisher's exact
test between conditions, which is the smallest honest comparison: the counts
are small, the runs are independent, and nothing here assumes a distribution.

TRAINING-TIME ANALYSIS of files written by `scripts/eval_part2.py`. It reads
reports, not the emulator.
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher's exact p for a/b successes against c/d.

    Summed over every table at most as probable as the observed one, which is
    the conventional two-sided definition and the one that does not need a
    normal approximation at these counts.
    """
    total, successes = b + d, a + c
    if min(b, d) == 0:
        return 1.0

    def probability(x: int) -> float:
        return comb(b, x) * comb(d, successes - x) / comb(total, successes)

    observed = probability(a)
    low, high = max(0, successes - d), min(b, successes)
    return min(1.0, sum(probability(x) for x in range(low, high + 1)
                        if probability(x) <= observed * (1 + 1e-9)))


def pool(paths, objective: str) -> tuple[int, int]:
    """(runs that reached the objective, runs) across several eval reports.

    The greedy run is one of the runs. `eval_part2` reports it separately
    because the design asks for it separately, but it is played under the same
    contract from the same state, and leaving it out of the denominator would
    flatter whichever side of a comparison happened to win it.
    """
    reached = runs = 0
    for path in paths:
        report = json.loads(Path(path).read_text())
        reached += int(report["reached"].get(objective, 0))
        runs += int(report["seeds"]) + 1
        if report["greedy"]["objectives"].get(objective) is not None:
            reached += 1
    return reached, runs


def table(conditions: dict, objective: str) -> dict:
    """Per-seed counts and the pooled count for each named condition."""
    rows = {}
    for name, paths in conditions.items():
        per_seed = [pool([path], objective)[0] for path in paths]
        reached, runs = pool(paths, objective)
        rows[name] = {"per_seed": per_seed, "pooled": [reached, runs],
                      "reports": [str(path) for path in paths]}
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", action="append", default=[], metavar="NAME=GLOB",
                        help="a named condition and a glob over its eval reports; "
                             "repeat once per condition, in the order to compare")
    parser.add_argument("--objective", default="get_parcel")
    parser.add_argument("--out", default="results/part2_seed_spread.json")
    args = parser.parse_args()

    conditions = {}
    for item in args.condition:
        name, _, pattern = item.partition("=")
        conditions[name] = sorted(Path().glob(pattern))
    rows = table(conditions, args.objective)

    names = list(rows)
    comparisons = []
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            first, second = rows[names[left]], rows[names[right]]
            comparisons.append({
                "between": [names[left], names[right]],
                "p": round(fisher_two_sided(*first["pooled"], *second["pooled"]), 5),
            })

    report = {
        "objective": args.objective,
        "conditions": rows,
        "fisher_two_sided": comparisons,
        "what": "Each condition is several TRAINING seeds of the same dataset, "
                "each scored over ten sampling seeds and a greedy run.",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
