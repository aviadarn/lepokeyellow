"""Can the frozen encoder say WHERE IN THE ROOM the player is standing?

The earlier probe answered this on one 467-frame route and said no: map id was
93.6% recoverable while position within a single map scored R^2 -0.297. That
is a small sample from one trajectory, and the whole diagnosis of this project
rests on it, so it is worth re-running against a proper dataset -- 31,759
frames with the player's (map, y, x) recorded for every one of them.

Probes y and x separately, per map, with the controls that make a probe score
mean something:

  shuffled labels  must collapse to chance, or the split leaks
  raw pixels       the same probe on downsampled screens; if raw pixels match
                   the encoder, the representation is not doing the work
  nearest tile     mean absolute error in TILES, which is the number that
                   decides whether a planner could use this

Splits are by EPISODE, not by frame. Frames from one episode are a continuous
walk, so splitting them randomly would put a step and its neighbour on opposite
sides and measure autocorrelation.

  uv run python -m scripts.probe_location checkpoints/yellow_v2/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch as t

from poke.goals import default_device
from poke.jepa import JEPA
from poke.maps import name as map_name
from poke.probe_linear import fit_ridge, predict, r2
from scripts.probe_position import downsample, encode_frames


def episode_of_frame(episodes: np.ndarray, n: int) -> np.ndarray:
    owner = np.zeros(n, dtype=np.int64)
    for index, (start, length) in enumerate(episodes):
        owner[int(start) : int(start) + int(length)] = index
    return owner


def group_split(groups: np.ndarray, folds: int, seed: int):
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    chunks = np.array_split(unique, folds)
    for held in chunks:
        test = np.isin(groups, held)
        yield np.flatnonzero(~test), np.flatnonzero(test)


def probe(x, y, groups, alpha, folds, seed):
    predictions = np.full(len(y), np.nan)
    for train, test in group_split(groups, folds, seed):
        if len(train) < x.shape[1] or not len(test):
            continue
        weights, intercept = fit_ridge(x[train], y[train], alpha)
        predictions[test] = predict(x[test], weights, intercept)
    ok = ~np.isnan(predictions)
    return r2(y[ok], predictions[ok]), float(np.abs(predictions[ok] - y[ok]).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("--data", default="episodes/yellow_v3.h5")
    parser.add_argument("--alpha", type=float, nargs="+", default=[1.0, 10.0, 100.0, 1000.0])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=8000)
    parser.add_argument("--min-frames", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/probe_location.json")
    parser.add_argument("--device", default=default_device())
    args = parser.parse_args()

    with h5py.File(args.data, "r") as file:
        if "location" not in file:
            raise SystemExit(
                f"{args.data} has no location dataset; collect with the current "
                "scripts/collect.py (schema 3)."
            )
        location = file["location"][:]
        episodes = file["episodes"][:]
        total = file["frames"].shape[0]
        owner = episode_of_frame(episodes, total)

        device = t.device(args.device)
        checkpoint = t.load(args.checkpoint, map_location=device, weights_only=False)
        model = JEPA(num_frames=checkpoint["config"]["num_frames"]).to(device)
        model.load_state_dict(checkpoint["jepa"])
        model.eval()

        rng = np.random.default_rng(args.seed)
        report = {}
        print(f"{'map':<32}{'axis':>5}{'encoder R2':>12}{'MAE tiles':>11}"
              f"{'pixels R2':>11}{'shuffled R2':>13}")
        for map_id in sorted(set(location[:, 0].tolist())):
            rows = np.flatnonzero((location[:, 0] == map_id) & (location[:, 1] >= 0))
            if len(rows) < args.min_frames:
                continue
            if len(rows) > args.max_frames:
                rows = np.sort(rng.choice(rows, args.max_frames, replace=False))

            frames = file["frames"][:, :, :, 0][rows]
            embeddings = encode_frames(model, frames, device).astype(np.float64)
            pixels = downsample(frames)
            groups = owner[rows]

            for axis, label in ((1, "y"), (2, "x")):
                target = location[rows, axis].astype(np.float64)
                if target.std() < 1e-6:
                    continue
                shuffled = target.copy()
                rng.shuffle(shuffled)
                best = max(
                    (probe(embeddings, target, groups, a, args.folds, args.seed)
                     for a in args.alpha),
                    key=lambda s: s[0],
                )
                pixel = max(
                    (probe(pixels, target, groups, a, args.folds, args.seed)
                     for a in args.alpha),
                    key=lambda s: s[0],
                )
                shuf = max(
                    (probe(embeddings, shuffled, groups, a, args.folds, args.seed)
                     for a in args.alpha),
                    key=lambda s: s[0],
                )
                report[f"map{map_id}_{label}"] = {
                    "frames": int(len(rows)),
                    "encoder_r2": best[0], "encoder_mae_tiles": best[1],
                    "pixels_r2": pixel[0], "shuffled_r2": shuf[0],
                    "target_std": float(target.std()),
                }
                print(f"{map_name(map_id)[:30]:<32}{label:>5}{best[0]:>12.3f}"
                      f"{best[1]:>11.2f}{pixel[0]:>11.3f}{shuf[0]:>13.3f}")

    good = [v for v in report.values() if v["encoder_r2"] > 0.3]
    leaks = [k for k, v in report.items() if v["shuffled_r2"] > 0.1]
    print()
    if leaks:
        print(f"LEAK: shuffled labels score above chance on {leaks}; "
              "the split is not clean.")
    elif not good:
        print("VERDICT: within-room position is NOT recoverable on any map. "
              "The encoder represents which room you are in and nothing "
              "finer, confirmed on a full dataset rather than one route. A "
              "planner has no signal to move the player across a room, and "
              "that is a training-signal problem, not a planner one.")
    else:
        print(f"VERDICT: within-room position IS recoverable on "
              f"{len(good)}/{len(report)} map-axis pairs. The earlier "
              f"single-route result understated it; re-examine the planner "
              f"before changing the training signal.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "data": args.data,
        "folds": args.folds,
        "alpha_sweep": args.alpha,
        "results": report,
    }, indent=2) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
