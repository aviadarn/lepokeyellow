"""Can the frozen encoder tell you WHERE along the route you are?

The play runs fail in a specific way: one-step prediction is excellent (44x to
631x better than a no-change baseline) and the planner still cannot follow the
route, because latent distance mislocalises -- unbounded re-anchoring matched
Pallet Town to rung 44 of 46 while the party was empty.

Two causes, opposite fixes:

  the information is in the embedding and Euclidean distance in a SIGReg-shaped
  space is simply not meaningful at range -> replace the planner's cost with
  something read out of the embedding, and the model stays as it is;

  the information is not in the embedding at all, because next-step prediction
  never required the encoder to represent route position -> no planner change
  helps, and the training signal has to change.

A linear probe on the frozen encoder separates them. The labels are free: the
frame's index along the demonstration IS its route position.

Three controls, because a probe score alone means nothing:

  shuffled labels   must collapse to ~0. If it does not, the split is leaking.
  raw pixels        the same probe on downsampled screens. If raw pixels do as
                    well, the encoder is adding nothing and the "representation"
                    is not doing any work.
  predict the mean  R^2 = 0 by construction, the floor every number is against.

Splits are contiguous blocks with a purge gap. Frames 60 emulator frames apart
are nearly identical, so a random split would put a frame's own neighbours on
both sides and measure autocorrelation instead of generalisation.

  uv run python -m scripts.probe_position checkpoints/yellow_v2/best.pt
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
from poke.ladder import demo_episode
from poke.probe_linear import cross_val


def encode_frames(model, frames: np.ndarray, device, batch: int = 128) -> np.ndarray:
    out = []
    with t.inference_mode():
        for start in range(0, len(frames), batch):
            chunk = frames[start : start + batch].astype(np.float32) / 127.5 - 1.0
            tensor = t.from_numpy(chunk).unsqueeze(1).to(device)
            out.append(model.encode(tensor).cpu().numpy())
    return np.concatenate(out)


def downsample(frames: np.ndarray, factor: int = 8) -> np.ndarray:
    """Block-mean pooling, as the raw-pixel control's features."""
    n, h, w = frames.shape
    trimmed = frames[:, : h // factor * factor, : w // factor * factor]
    pooled = trimmed.reshape(n, h // factor, factor, w // factor, factor).mean(axis=(2, 4))
    return pooled.reshape(n, -1).astype(np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("--data", default="episodes/pruned_demos.h5")
    parser.add_argument("--alpha", type=float, nargs="+",
                        default=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
                        help="ridge strengths to sweep; the best per arm is "
                             "reported, so a verdict cannot rest on one "
                             "arbitrary setting")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--purge", type=int, default=10)
    parser.add_argument("--blocks", type=int, default=40,
                        help="contiguous chunks dealt round-robin to the folds; "
                             "must exceed folds or the probe is asked to "
                             "extrapolate along the route")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--by-map", action="store_true",
                        help="also ask whether the probe is an odometer or just "
                             "a map detector: map-id accuracy, and position "
                             "within a single map")
    parser.add_argument("--rom", default="pokeyellow.gbc")
    parser.add_argument("--route", default="routes/yellow_pikachu.json")
    parser.add_argument("--state", default="states/bedroom.state")
    parser.add_argument("--out", default="results/probe_position.json")
    parser.add_argument("--device", default=default_device())
    args = parser.parse_args()

    # One demonstration episode. The others are deterministic replays of the
    # same route, so including them would quadruple the rows without adding a
    # single new screen.
    start, length, name = demo_episode(args.data)
    with h5py.File(args.data, "r") as file:
        frames = file["frames"][start : start + length, :, :, 0]
        maps = None  # the dataset stores no map id; position is the label here
    position = np.arange(length, dtype=np.float64)

    device = t.device(args.device)
    checkpoint = t.load(args.checkpoint, map_location=device, weights_only=False)
    model = JEPA(num_frames=checkpoint["config"]["num_frames"]).to(device)
    model.load_state_dict(checkpoint["jepa"])
    model.eval()

    embeddings = encode_frames(model, frames, device).astype(np.float64)
    pixels = downsample(frames)
    rng = np.random.default_rng(args.seed)
    shuffled = position.copy()
    rng.shuffle(shuffled)

    print(f"demo {name!r}: {length} frames, embedding {embeddings.shape[1]}d, "
          f"raw-pixel control {pixels.shape[1]}d")
    print(f"blocked {args.folds}-fold CV over {args.blocks} interleaved blocks, "
          f"purge {args.purge}, ridge alpha {args.alpha}\n")

    arms = {
        "encoder": (embeddings, position),
        "raw pixels": (pixels, position),
        "encoder, shuffled labels": (embeddings, shuffled),
    }
    results = {}
    print(f"{'probe':<26}{'R^2':>8}{'MAE (actions)':>16}{'best alpha':>12}")
    for label, (x, y) in arms.items():
        swept = {}
        for alpha in args.alpha:
            scored = cross_val(x, y, alpha=alpha, folds=args.folds,
                               purge=args.purge, blocks=args.blocks)
            swept[alpha] = {k: v for k, v in scored.items() if k != "predictions"}
        # Pick on MAE: R^2 is squared error and a handful of large misses can
        # drive it negative while typical error is plainly better than chance,
        # which is exactly what this encoder does.
        best_alpha = min(swept, key=lambda a: swept[a]["mae"])
        results[label] = {"best_alpha": best_alpha, "sweep": swept, **swept[best_alpha]}
        print(f"{label:<26}{swept[best_alpha]['r2']:>8.3f}"
              f"{swept[best_alpha]['mae']:>16.1f}{best_alpha:>12g}")
    print(f"{'predict the mean':<26}{0.0:>8.3f}{float(np.abs(position - position.mean()).mean()):>16.1f}")

    by_map = {}
    if args.by_map:
        # Replaying the route is the only way to label each frame with its map;
        # the dataset does not store one.
        from poke.actions import ACTIONS
        from poke.env import PokeEnv
        from poke.ram import cur_map
        from poke.routes import Route

        route = Route.load(args.route)
        env = PokeEnv(args.rom, args.state, headless=True)
        try:
            env.reset()
            maps = [cur_map(env)]
            for action in route.actions:
                env.step(ACTIONS.index(action))
                maps.append(cur_map(env))
        finally:
            env.close()
        # The episode runs past the route by --post-goal frames, which all sit
        # on the map the route ended in.
        while len(maps) < length:
            maps.append(maps[-1])
        maps = np.array(maps[:length])

        print("\nis it an odometer, or a map detector?")
        # Map identity, one-vs-rest ridge, scored as accuracy.
        ids = sorted(set(maps.tolist()))
        votes = np.zeros((length, len(ids)))
        for column, map_id in enumerate(ids):
            target = (maps == map_id).astype(np.float64)
            scored = cross_val(embeddings, target, alpha=100.0, folds=args.folds,
                               purge=args.purge, blocks=args.blocks)
            votes[:, column] = scored["predictions"]
        predicted_map = np.array(ids)[votes.argmax(axis=1)]
        accuracy = float((predicted_map == maps).mean())
        majority = float(max((maps == i).mean() for i in ids))
        print(f"  map id from the embedding: {accuracy:.1%} accurate "
              f"(majority-class baseline {majority:.1%}, {len(ids)} maps)")
        by_map["map_id_accuracy"] = accuracy
        by_map["map_id_majority_baseline"] = majority

        # Position WITHIN the single longest map stretch. If the probe is an
        # odometer this still works; if it is a map detector it collapses.
        best = max(ids, key=lambda i: int((maps == i).sum()))
        mask = maps == best
        inside_x = embeddings[mask]
        inside_y = position[mask]
        inner_blocks = max(args.folds, min(args.blocks, len(inside_y) // 8))
        inner = cross_val(inside_x, inside_y, alpha=100.0, folds=args.folds,
                          purge=args.purge, blocks=inner_blocks)
        print(f"  position within map {best} alone ({int(mask.sum())} frames): "
              f"R^2 {inner['r2']:.3f}, +/-{inner['mae']:.0f} actions")
        by_map["largest_map"] = int(best)
        by_map["largest_map_frames"] = int(mask.sum())
        by_map["within_map_r2"] = inner["r2"]
        by_map["within_map_mae"] = inner["mae"]
        by_map["is_map_detector"] = bool(accuracy > 0.8 and inner["r2"] < 0.2)

    encoder_r2 = results["encoder"]["r2"]
    encoder_mae = results["encoder"]["mae"]
    pixel_mae = results["raw pixels"]["mae"]
    shuffled_mae = results["encoder, shuffled labels"]["mae"]
    mean_mae = float(np.abs(position - position.mean()).mean())

    print()
    beats_chance = encoder_mae < min(shuffled_mae, mean_mae)
    beats_pixels = encoder_mae < pixel_mae
    if shuffled_mae < mean_mae * 0.9:
        print("LEAK: shuffled labels beat the mean; the split is not clean. "
              "Nothing else here can be trusted.")
    elif not beats_chance:
        print("VERDICT: route position is NOT linearly recoverable from the "
              "frozen encoder -- it does no better than chance. No change to "
              "the planner's cost will fix localisation; the training signal "
              "has to change.")
    elif not beats_pixels:
        print("VERDICT: the encoder does no better than downsampled raw "
              "pixels. Position is recoverable, but not because of the "
              "representation.")
    elif encoder_r2 < 0.2:
        print(f"VERDICT: WEAK. Typical error {encoder_mae:.0f} actions against "
              f"{mean_mae:.0f} for the mean and {pixel_mae:.0f} for raw "
              f"pixels, so the encoder does carry positional information -- "
              f"but R^2 is {encoder_r2:.2f}, at or below chance, so the errors "
              f"are heavy-tailed: mostly reasonable, occasionally far wrong. "
              f"That is enough to explain the planner matching Pallet Town to "
              f"rung 44 of 46, and not enough to use as a progress signal. "
              f"Reading progress out of the embedding needs a better readout "
              f"than a linear one, or a training signal that makes position "
              f"linearly available.")
    elif by_map.get("is_map_detector"):
        print(f"VERDICT: MAP DETECTOR, not an odometer. Route position scores "
              f"R^2 {encoder_r2:.3f} overall, but that is carried by map "
              f"identity: map id is {by_map['map_id_accuracy']:.1%} accurate "
              f"against a {by_map['map_id_majority_baseline']:.1%} "
              f"majority-class baseline, while position WITHIN map "
              f"{by_map['largest_map']} alone scores "
              f"{by_map['within_map_r2']:.3f} over "
              f"{by_map['largest_map_frames']} frames -- at or below chance. "
              f"The encoder knows which room you are in and not where in the "
              f"room, and +/-{encoder_mae:.0f} actions is about the length of "
              f"a within-map stretch. Progress read this way is not a usable "
              f"planning objective: maximising it walks to the highest-scoring "
              f"map and stays there.")
    else:
        print(f"VERDICT: route position IS linearly recoverable (R^2 "
              f"{encoder_r2:.3f}, +/-{encoder_mae:.0f} actions), beating raw "
              f"pixels ({pixel_mae:.0f}). The information is there and "
              f"Euclidean distance is the wrong way to read it -- replace the "
              f"planner's cost with predicted progress; the model can stay. "
              f"Re-run with --by-map before trusting this: a probe that is "
              f"really a map detector scores well here too.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "epoch": checkpoint.get("epoch"),
        "data": args.data,
        "episode": name,
        "frames": int(length),
        "embedding_dims": int(embeddings.shape[1]),
        "pixel_dims": int(pixels.shape[1]),
        "folds": args.folds,
        "blocks": args.blocks,
        "purge": args.purge,
        "alpha_sweep": args.alpha,
        "results": results,
        "mean_baseline_mae": float(np.abs(position - position.mean()).mean()),
        "by_map": by_map,
    }, indent=2) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
