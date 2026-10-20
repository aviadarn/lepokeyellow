"""How far, in tiles, does a k-step rollout land from where the game lands?

The planner never sees a frame of its plan. It scores an action sequence by
decoding PREDICTED embeddings into tiles, so what decides whether it can
navigate is not the predictor's MSE in latent space and not the probe's error
on real screens -- it is the composition of the two. This measures that
composition against the emulator's own record of where the player was.

Three errors, separated:

  head      decode(encode(real frame at i+k)) vs the recorded tile. What the
            read-out costs.
  rollout   decode(rollout(encode(frame i), recorded actions)) vs the same
            recorded tile. Head error plus everything the predictor drifts.
  map       how often the rollout still puts the player in the right room.

Coordinates are read with the head of the map the player is REALLY on, so a
misclassified room shows up in the map column rather than as a nonsense tile
distance between two different coordinate frames.

The probe is fitted on a disjoint set of episodes from the ones measured, so
these are held-out numbers.

`--context` seeds the rollout with that many real frames instead of one.
Training gave the predictor num_frames-1 frames of context at every step while
`JEPA.predict_rollout` gives it one, and if that mismatch is what the planner
is paying for, it costs nothing to stop paying it.

  uv run python -m scripts.rollout_tiles checkpoints/yellow_v2/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch as t

from poke import coords_probe
from poke.goals import default_device
from poke.jepa import JEPA
from poke.maps import name as map_name
from poke.rollout import rollout_from_context
from poke.splits import frame_episodes


def sample_starts(location, owner, episodes, allowed, context, horizon,
                  count, rng):
    """Start indices with `context` real frames behind and `horizon` ahead,
    all inside one episode and all with a readable position."""
    starts = []
    for index in allowed:
        begin, length = int(episodes[index][0]), int(episodes[index][1])
        first = begin + context - 1
        last = begin + length - horizon - 1
        if last < first:
            continue
        starts.append(np.arange(first, last + 1))
    if not starts:
        raise SystemExit(
            f"no episode is long enough for {context} frames of context and a "
            f"{horizon}-step horizon"
        )
    starts = np.concatenate(starts)
    readable = np.array([
        bool((location[i - context + 1 : i + horizon + 1, 0] >= 0).all())
        for i in starts
    ])
    starts = starts[readable]
    if len(starts) > count:
        starts = np.sort(rng.choice(starts, count, replace=False))
    return starts


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("checkpoint")
    parser.add_argument("--data", default="episodes/yellow_v3.h5")
    parser.add_argument("--head", default="mlp", choices=("linear", "mlp"))
    parser.add_argument("--fit-on", default="real",
                        choices=("real", "predicted", "both"),
                        help="fit the coordinate head on real encodings, on "
                             "predicted embeddings (what the planner decodes), "
                             "or on both")
    parser.add_argument("--fit-steps", type=int, default=2,
                        help="rollout length used to build predicted features")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--context", type=int, nargs="+", default=[1, 3],
                        help="real frames seeding the rollout; 1 is what the "
                             "planner does today")
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--maps", type=int, nargs="+",
                        help="restrict the report to these map ids")
    parser.add_argument("--holdout", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/rollout_tiles.json")
    parser.add_argument("--device", default=default_device())
    args = parser.parse_args()

    device = t.device(args.device)
    checkpoint = t.load(args.checkpoint, map_location=device, weights_only=False)
    model = JEPA(num_frames=checkpoint["config"]["num_frames"]).to(device)
    model.load_state_dict(checkpoint["jepa"])
    model.eval()

    rng = np.random.default_rng(args.seed)
    with h5py.File(args.data, "r") as file:
        episodes = file["episodes"][:]
        location = file["location"][:]
        actions = file["actions"][:].astype(np.float32)
        owner = frame_episodes(episodes, file["frames"].shape[0])

        order = rng.permutation(len(episodes))
        cut = max(1, int(round(len(episodes) * args.holdout)))
        test_episodes, train_episodes = order[:cut], order[cut:]

        probe = coords_probe.fit(model, args.data, device, head=args.head,
                                 only_episodes=train_episodes,
                                 seed=args.seed, on=args.fit_on,
                                 steps=args.fit_steps)
        torch_probe = probe.to_torch(device)
        map_of_id = {map_id: index for index, map_id in enumerate(probe.maps)}
        print(f"{probe.head} head fitted on {probe.fitted_on} embeddings from "
              f"{len(train_episodes)} episodes, measured on {len(test_episodes)}")
        if probe.holdout_tile_mae:
            print("  its own held-out error: " + "  ".join(
                f"{k}={v:.2f}" for k, v in sorted(probe.holdout_tile_mae.items())))

        report = {}
        for context in args.context:
            starts = sample_starts(location, owner, episodes, test_episodes,
                                   context, args.horizon, args.samples, rng)
            # Every frame the measurement needs: the context behind each start
            # and the horizon ahead of it.
            offsets = np.arange(-(context - 1), args.horizon + 1)
            wanted = np.unique((starts[:, None] + offsets[None, :]).ravel())
            frames = file["frames"][:, :, :, 0][wanted]
            position = {index: row for index, row in enumerate(wanted)}
            lookup = {int(index): row for row, index in enumerate(wanted)}

            with t.inference_mode():
                chunks = []
                for begin in range(0, len(frames), 256):
                    block = frames[begin : begin + 256].astype(np.float32) / 127.5 - 1.0
                    chunks.append(model.encode(t.from_numpy(block).unsqueeze(1).to(device)))
                embeddings = t.cat(chunks)

            seed_rows = np.array([[lookup[int(i + d)] for d in range(-(context - 1), 1)]
                                  for i in starts])
            seed = embeddings[t.from_numpy(seed_rows).to(device)]
            action_rows = np.array([
                np.arange(i - context + 1, i + args.horizon) for i in starts
            ])
            action_batch = t.from_numpy(actions[action_rows]).to(device)

            with t.inference_mode():
                predicted = rollout_from_context(model, seed, action_batch)

            truth = np.stack([location[starts + k] for k in range(1, args.horizon + 1)],
                             axis=1)                       # [N, H, 3]
            real_map_index = t.from_numpy(
                np.vectorize(lambda m: map_of_id.get(int(m), 0))(truth[:, :, 0])
            ).to(device)

            with t.inference_mode():
                _, py, px = coords_probe.decode(predicted, torch_probe,
                                                map_index=real_map_index)
                guessed_map, _, _ = coords_probe.decode(predicted, torch_probe)
                real_rows = np.array([[lookup[int(i + k)]
                                       for k in range(1, args.horizon + 1)]
                                      for i in starts])
                head_states = embeddings[t.from_numpy(real_rows).to(device)]
                _, hy, hx = coords_probe.decode(head_states, torch_probe,
                                                map_index=real_map_index)
                # The baseline the predictor has to beat to be worth having:
                # decode the CURRENT frame and claim the player has not moved.
                # If a one-step rollout is further from the truth than that,
                # the predictor is actively destroying position information and
                # no read-out can be fitted around it.
                stay_rows = np.array([[lookup[int(i)]] * args.horizon
                                      for i in starts])
                stay_states = embeddings[t.from_numpy(stay_rows).to(device)]
                _, sy, sx = coords_probe.decode(stay_states, torch_probe,
                                                map_index=real_map_index)

            truth_y = t.from_numpy(truth[:, :, 1].astype(np.float32)).to(device)
            truth_x = t.from_numpy(truth[:, :, 2].astype(np.float32)).to(device)
            # The coordinate frames only line up inside one map, so tile error
            # is measured where the player did not change room during the
            # horizon; room changes are reported as map accuracy instead.
            same_room = t.from_numpy(
                (truth[:, :, 0] == location[starts][:, None, 0])
            ).to(device)
            selected = (t.from_numpy(np.isin(truth[:, :, 0], args.maps)).to(device)
                        & same_room) if args.maps else same_room

            rows = []
            for k in range(args.horizon):
                keep = selected[:, k]
                n = int(keep.sum())
                if not n:
                    continue
                tiles = lambda a, b: float(
                    ((a[:, k][keep] - truth_y[:, k][keep]).abs()
                     + (b[:, k][keep] - truth_x[:, k][keep]).abs()).mean()
                )
                rows.append({
                    "k": k + 1,
                    "n": n,
                    "head_tiles": tiles(hy, hx),
                    "stay_tiles": tiles(sy, sx),
                    "rollout_tiles": tiles(py, px),
                    "map_accuracy": float(
                        (guessed_map[:, k][keep] == real_map_index[:, k][keep])
                        .float().mean()
                    ),
                })
            report[f"context{context}"] = rows

            print(f"\ncontext {context} frame(s), {len(starts)} starts"
                  f"{'' if not args.maps else ' on maps ' + str(args.maps)}")
            print(f"{'k':>3}{'n':>7}{'head':>9}{'stay':>9}{'rollout':>9}"
                  f"{'map ok':>9}")
            for row in rows:
                print(f"{row['k']:>3}{row['n']:>7}{row['head_tiles']:>9.2f}"
                      f"{row['stay_tiles']:>9.2f}{row['rollout_tiles']:>9.2f}"
                      f"{row['map_accuracy']:>9.1%}")

    first = {name: rows[0] for name, rows in report.items() if rows}
    print()
    for name, row in first.items():
        print(f"{name}: one step costs {row['rollout_tiles']:.2f} tiles against "
              f"the head's own {row['head_tiles']:.2f} and {row['stay_tiles']:.2f} "
              f"for assuming no movement")
        if row["rollout_tiles"] > row["stay_tiles"]:
            print("  VERDICT: the predictor is WORSE than assuming the player "
                  "did not move, so it is destroying position rather than "
                  "predicting it. No read-out fixes that; the training signal "
                  "has to.")
    if len(first) > 1:
        best = min(first.items(), key=lambda kv: kv[1]["rollout_tiles"])
        worst = max(first.items(), key=lambda kv: kv[1]["rollout_tiles"])
        if worst[1]["rollout_tiles"] - best[1]["rollout_tiles"] > 0.25:
            print(f"VERDICT: {best[0]} is {worst[1]['rollout_tiles'] - best[1]['rollout_tiles']:.2f} "
                  f"tiles better at one step. The planner seeds its rollout "
                  f"with one frame; seeding it the way training did is free.")
        else:
            print("VERDICT: context length does not change one-step tile error, "
                  "so the training/planning mismatch is not what the planner "
                  "is paying for.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "data": args.data,
        "head": probe.head,
        "fitted_on": probe.fitted_on,
        "holdout_episodes": int(cut),
        "samples": args.samples,
        "maps": args.maps,
        "probe_holdout_tile_mae": probe.holdout_tile_mae,
        "results": report,
    }, indent=2) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
