"""Does a non-linear head read the player's tile out of the encoder better?

The planner stalls in Pallet Town because the LINEAR coordinate probe's error
there (1.44 tiles in x, 1.88 in y) is larger than the one-hop granularity of
the cost field it is scoring against. Two explanations fit that: the encoder
has thrown the information away, or a linear read-out cannot extract it.

The second is favoured by a measurement already in results/probe_location.json:
raw 8x-downsampled pixels, under the same linear probe, score R^2 0.912/0.803
on Pallet against the encoder's 0.760/0.655. A linear map of raw greyscale
should not beat a linear map of a learned representation if the representation
had dropped position. So this script keeps the frozen encoder and the frozen
dataset fixed and changes only the head.

Same protocol as scripts/probe_location.py, so the numbers are comparable:

  splits          by EPISODE, never by frame -- consecutive frames are one walk
  linear          ridge with an alpha sweep, the incumbent
  mlp             two hidden layers, early-stopped on held-out episodes carved
                  out of each fold's TRAINING set only
  mlp, shuffled   the control that matters most here: an MLP has the capacity
                  to memorise, so if shuffled labels score above chance the
                  split leaks and every other number is void

Decision rule, stated before running: the planner needs error below the one
tile that separates adjacent BFS levels, so the MLP is worth wiring into the
planner only if Pallet's MAE drops under about 1.0 on BOTH axes.

  uv run python -m scripts.probe_nonlinear checkpoints/yellow_v2/best.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch as t

from poke.goals import default_device
from poke.jepa import JEPA
from poke.maps import name as map_name
from poke.probe_linear import fit_ridge, predict, r2
from poke.probe_mlp import fit_mlp
from poke.splits import frame_episodes
from scripts.probe_location import group_split
from scripts.probe_position import downsample, encode_frames


def score(truth: np.ndarray, guess: np.ndarray) -> tuple[float, float]:
    ok = ~np.isnan(guess)
    return r2(truth[ok], guess[ok]), float(np.abs(guess[ok] - truth[ok]).mean())


def linear_probe(x, y, groups, alphas, folds, seed):
    best = None
    for alpha in alphas:
        predictions = np.full(len(y), np.nan)
        for train, test in group_split(groups, folds, seed):
            if len(train) < x.shape[1] or not len(test):
                continue
            weights, intercept = fit_ridge(x[train], y[train], alpha)
            predictions[test] = predict(x[test], weights, intercept)
        candidate = (*score(y, predictions), alpha)
        if best is None or candidate[0] > best[0]:
            best = candidate
    return best


def mlp_probe(x, y, groups, folds, seed, device, **kwargs):
    predictions = np.full(len(y), np.nan)
    epochs = []
    for train, test in group_split(groups, folds, seed):
        if not len(test) or len(train) < 64:
            continue
        head = fit_mlp(
            x[train], y[train], groups=groups[train], seed=seed,
            device=device, **kwargs,
        )
        predictions[test] = head.predict(x[test])
        epochs.append(head.epochs_used)
    return (*score(y, predictions), float(np.mean(epochs)) if epochs else 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("checkpoint")
    parser.add_argument("--data", default="episodes/yellow_v3.h5")
    parser.add_argument("--alpha", type=float, nargs="+",
                        default=[1.0, 10.0, 100.0, 1000.0])
    parser.add_argument("--maps", type=int, nargs="+",
                        help="only these map ids (default: every map with data)")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=8000)
    parser.add_argument("--min-frames", type=int, default=400)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--pixels", action="store_true",
                        help="also probe raw pixels with the MLP; a bound on "
                             "what any head could recover, not usable for "
                             "planning (the model predicts embeddings)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/probe_nonlinear.json")
    parser.add_argument("--device", default=default_device())
    args = parser.parse_args()

    device = t.device(args.device)
    checkpoint = t.load(args.checkpoint, map_location=device, weights_only=False)
    model = JEPA(num_frames=checkpoint["config"]["num_frames"]).to(device)
    model.load_state_dict(checkpoint["jepa"])
    model.eval()

    mlp_kwargs = dict(hidden=args.hidden, depth=args.depth, epochs=args.epochs,
                      patience=args.patience, weight_decay=args.weight_decay)

    report = {}
    with h5py.File(args.data, "r") as file:
        if "location" not in file:
            raise SystemExit(f"{args.data} has no location dataset (schema 3)")
        location = file["location"][:]
        owner = frame_episodes(file["episodes"][:], file["frames"].shape[0])
        rng = np.random.default_rng(args.seed)

        maps = args.maps or sorted(set(location[:, 0].tolist()))
        print(f"{'map':<26}{'axis':>5}{'linear R2':>11}{'linear MAE':>12}"
              f"{'mlp R2':>9}{'mlp MAE':>9}{'shuffled':>10}{'epochs':>8}")
        for map_id in maps:
            rows = np.flatnonzero((location[:, 0] == map_id) & (location[:, 1] >= 0))
            if len(rows) < args.min_frames:
                continue
            if len(rows) > args.max_frames:
                rows = np.sort(rng.choice(rows, args.max_frames, replace=False))

            frames = file["frames"][:, :, :, 0][rows]
            embeddings = encode_frames(model, frames, device).astype(np.float64)
            pixels = downsample(frames) if args.pixels else None
            groups = owner[rows]

            for axis, label in ((1, "y"), (2, "x")):
                target = location[rows, axis].astype(np.float64)
                if target.std() < 1e-6:
                    continue
                started = time.time()
                lin_r2, lin_mae, alpha = linear_probe(
                    embeddings, target, groups, args.alpha, args.folds, args.seed
                )
                mlp_r2, mlp_mae, epochs = mlp_probe(
                    embeddings, target, groups, args.folds, args.seed, device,
                    **mlp_kwargs
                )
                shuffled = target.copy()
                rng.shuffle(shuffled)
                shuf_r2, _, _ = mlp_probe(
                    embeddings, shuffled, groups, args.folds, args.seed, device,
                    **mlp_kwargs
                )
                entry = {
                    "frames": int(len(rows)),
                    "linear_r2": lin_r2, "linear_mae_tiles": lin_mae,
                    "linear_alpha": alpha,
                    "mlp_r2": mlp_r2, "mlp_mae_tiles": mlp_mae,
                    "mlp_shuffled_r2": shuf_r2,
                    "mlp_mean_epochs": epochs,
                    "seconds": round(time.time() - started, 1),
                }
                if pixels is not None:
                    pr2, pmae, palpha = linear_probe(
                        pixels, target, groups, args.alpha, args.folds, args.seed
                    )
                    mr2, mmae, _ = mlp_probe(
                        pixels, target, groups, args.folds, args.seed, device,
                        **mlp_kwargs
                    )
                    entry |= {"pixels_linear_r2": pr2,
                              "pixels_linear_mae_tiles": pmae,
                              "pixels_linear_alpha": palpha,
                              "pixels_mlp_r2": mr2,
                              "pixels_mlp_mae_tiles": mmae}
                report[f"map{map_id}_{label}"] = entry
                print(f"{map_name(map_id)[:24]:<26}{label:>5}{lin_r2:>11.3f}"
                      f"{lin_mae:>12.2f}{mlp_r2:>9.3f}{mlp_mae:>9.2f}"
                      f"{shuf_r2:>10.3f}{epochs:>8.0f}")

    leaks = [k for k, v in report.items() if v["mlp_shuffled_r2"] > 0.1]
    pallet = {k: v for k, v in report.items() if k.startswith("map0_")}
    print()
    if leaks:
        print(f"LEAK: shuffled labels score above chance on {leaks}. The MLP is "
              "reading something other than the label; every number above is void.")
    elif pallet:
        worst = max(v["mlp_mae_tiles"] for v in pallet.values())
        was = max(v["linear_mae_tiles"] for v in pallet.values())
        print(f"Pallet Town worst-axis MAE: linear {was:.2f} -> mlp {worst:.2f} tiles")
        if worst < 1.0:
            print("VERDICT: below the one-tile granularity of the BFS cost "
                  "field. Wire the MLP head into the planner and re-run "
                  "scripts/nav_bench.py.")
        else:
            print("VERDICT: still above one tile, so the planner would still be "
                  "choosing between options it cannot tell apart. A better head "
                  "is not the fix; more Pallet Town data or a coordinate head "
                  "trained WITH the encoder is the next thing to try.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "data": args.data,
        "folds": args.folds,
        "alpha_sweep": args.alpha,
        "mlp": mlp_kwargs,
        "results": report,
    }, indent=2) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
