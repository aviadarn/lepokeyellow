"""Is the model wrong, or is the objective badly shaped?

A play run converged on pressing A: 291 of 501 actions, predicted cost down to
0.0051, player never out of the bedroom. Two very different things produce
that, and the fix differs completely:

  the model is wrong   it mispredicts what A does, so the planner is steering
                       with a broken map -> more data, more training
  the objective is bad  the model predicts A correctly, and a state reached by
                       pressing A genuinely sits closer to the goal embedding
                       than a state reached by walking -> goal distance is the
                       wrong thing to minimise, and no amount of training fixes
                       it

So measure both, against the real emulator, for every action from a given
state:

  prediction error  ||predicted next embedding - encoded real next screen||,
                    against the baseline of predicting no change at all. A
                    model that beats "copy the current embedding" knows
                    something; one that does not, does not.
  goal pull         how the goal distance moves, as the model BELIEVES and as
                    it ACTUALLY does. When the model thinks A helps and reality
                    agrees, A really is the greedy move and the objective is at
                    fault.

  uv run python -m scripts.action_check checkpoints/yellow_pikachu/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch as t

from poke.actions import ACTIONS, to_multihot
from poke.env import PokeEnv
from poke.goals import default_device
from poke.jepa import JEPA
from poke.ladder import build as build_ladder
from poke.ladder import dedupe as dedupe_ladder
from poke.ladder import load_frames as load_ladder_frames
from poke.ram import coords, cur_map, party_count

from scripts.play import to_tensor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("--rom", default="pokeyellow.gbc")
    parser.add_argument("--state", default="states/bedroom.state")
    parser.add_argument("--data", default="episodes/pruned_demos.h5")
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--rung", type=int, default=0, help="which rung to score against")
    parser.add_argument("--warmup", type=int, default=0,
                        help="actions of the demonstration to replay first")
    parser.add_argument("--act-frames", type=int, default=60)
    parser.add_argument("--out", default="results/action_check.json")
    parser.add_argument("--game")
    parser.add_argument("--device", default=default_device())
    args = parser.parse_args()

    device = t.device(args.device)
    checkpoint = t.load(args.checkpoint, map_location=device, weights_only=False)
    model = JEPA(num_frames=checkpoint["config"]["num_frames"]).to(device)
    model.load_state_dict(checkpoint["jepa"])
    model.eval()

    ladder = dedupe_ladder(args.data, build_ladder(args.data, args.stride))
    frames = load_ladder_frames(args.data, ladder)
    with t.inference_mode():
        goal = model.encode(
            t.from_numpy(frames[args.rung].astype(np.float32) / 127.5 - 1.0)
            .unsqueeze(0).unsqueeze(0).to(device)
        )

    env = PokeEnv(args.rom, args.state, headless=True,
                  act_frames=args.act_frames, game=args.game)
    rows = []
    try:
        env.reset()
        for index in range(args.warmup):
            env.step(ACTIONS.index("a"))
        here = env.snapshot()
        with t.inference_mode():
            current = model.encode(to_tensor(env.screen(), device))
        base_distance = float((current - goal).square().mean())
        start_position = (cur_map(env), *coords(env), party_count(env))

        for action_index, action in enumerate(ACTIONS):
            env.restore(here)
            with t.inference_mode():
                action_embedding = model.action_encoder(
                    t.from_numpy(to_multihot(action_index)).to(device).view(1, 1, -1)
                )
                predicted = model.predictor(current.unsqueeze(1), action_embedding)[:, -1]
            env.step(action_index)
            with t.inference_mode():
                actual = model.encode(to_tensor(env.screen(), device))

            rows.append({
                "action": action,
                "prediction_error": float((predicted - actual).square().mean()),
                "no_change_baseline": float((current - actual).square().mean()),
                "believed_distance": float((predicted - goal).square().mean()),
                "actual_distance": float((actual - goal).square().mean()),
                "moved": (cur_map(env), *coords(env), party_count(env)) != start_position,
                "position": [cur_map(env), *coords(env)],
            })
        env.restore(here)
    finally:
        env.close()

    print(f"state={args.state} warmup={args.warmup} rung={args.rung} "
          f"of {len(ladder)}  start={start_position}")
    print(f"goal distance from here: {base_distance:.4f}\n")
    header = (f"{'action':<7} {'pred err':>9} {'no-change':>10} {'believed':>9} "
              f"{'actual':>9} {'moved':>6}")
    print(header)
    for row in rows:
        print(f"{row['action']:<7} {row['prediction_error']:>9.4f} "
              f"{row['no_change_baseline']:>10.4f} {row['believed_distance']:>9.4f} "
              f"{row['actual_distance']:>9.4f} {str(row['moved']):>6}")

    believed = min(rows, key=lambda r: r["believed_distance"])
    truth = min(rows, key=lambda r: r["actual_distance"])

    # Only actions that change the screen can be scored against "predict no
    # change". Where the screen is byte-identical the baseline is exactly 0.0
    # and unbeatable, and counting those as losses understates the model: on
    # the first clean run it reported 4/7 while the predictor was beating the
    # baseline on every action that did anything.
    scored = [r for r in rows if r["no_change_baseline"] > 0.0]
    beats = [r for r in scored if r["prediction_error"] < r["no_change_baseline"]]
    print(f"\nmodel would pick: {believed['action']}   "
          f"actually closest: {truth['action']}")
    print(f"beats the no-change baseline on {len(beats)}/{len(scored)} actions "
          f"that change the screen "
          f"({len(rows) - len(scored)} leave it byte-identical, where the "
          f"baseline is exact)")
    if scored:
        ratios = [r["no_change_baseline"] / max(r["prediction_error"], 1e-12)
                  for r in scored]
        print(f"prediction error is {min(ratios):.0f}x to {max(ratios):.0f}x "
              f"smaller than the baseline on those")

    if believed["action"] != truth["action"]:
        print("VERDICT: the model picks an action that is not actually closest. "
              "Prediction error is doing the damage; more data or more training "
              "is the lever.")
    elif not truth["moved"]:
        print("VERDICT: the model is right and the greedy action does not move "
              "the player. Goal distance is the wrong objective here, and more "
              "training will not fix it.")
    else:
        print("VERDICT: the model is right and the greedy action moves the "
              "player. Neither the predictor nor the objective is at fault at "
              "this state; a failure to play is the search's or the horizon's.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "epoch": checkpoint.get("epoch"),
        "state": args.state,
        "warmup": args.warmup,
        "rung": args.rung,
        "rungs": len(ladder),
        "start_position": list(start_position),
        "goal_distance": base_distance,
        "scored_actions": len(scored),
        "beats_no_change_baseline": len(beats),
        "model_would_pick": believed["action"],
        "actually_closest": truth["action"],
        "rows": rows,
    }, indent=2) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
