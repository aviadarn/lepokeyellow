"""The repeatability test, plus the calibration the post leaves out.

The blog post runs 100 fresh planner seeds against a fixed model and start
state and reports 52/100. That number alone cannot distinguish a planner that
mostly works from one that has no idea which of its plans will work: CEM
returns a predicted cost for every plan, and if that cost does not separate
the plans that succeed from the ones that fail, the search is exploiting model
error rather than using model knowledge.

So each seed is planned, executed, and both numbers are kept. The AUC at the
end is the probability that a randomly chosen successful plan was predicted
cheaper than a randomly chosen failed one. 0.5 means the planner is blind.

  uv run python -m scripts.sweep_seeds checkpoints/rollout/best.pt --seeds 100
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch as t

from poke.actions import ACTIONS
from poke.baselines import run_episode
from poke.env import PokeEnv
from poke.games import resolve
from poke.goals import (
    check_dataset,
    default_device,
    goal_indices,
    load_frame,
    start_index,
)
from poke.jepa import JEPA
from poke.tasks import get as get_task

from scripts.plan_cem import cem_plan


def rank_auc(costs: np.ndarray, successes: np.ndarray) -> float | None:
    """AUC of "lower predicted cost implies success", by rank, ties at 0.5."""
    positive, negative = costs[successes], costs[~successes]
    if not len(positive) or not len(negative):
        return None
    # Lower cost should mean success, so score by negated cost.
    scores = np.concatenate([-positive, -negative])
    order = scores.argsort(kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks within tie groups.
    unique, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(unique))
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    n_pos = len(positive)
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * len(negative)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--rom", default="pokeyellow.gb")
    parser.add_argument("--task", default="yellow_pikachu")
    parser.add_argument("--data", default="episodes/poke.h5")
    parser.add_argument("--state")
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--elites", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--max-goals", type=int, default=8)
    parser.add_argument("--act-frames", type=int, default=60)
    parser.add_argument("--untrained", action="store_true",
                        help="randomly initialised predictor, the post's control")
    parser.add_argument("--out", default="results/sweep.json")
    parser.add_argument("--game")
    parser.add_argument("--device", default=default_device())
    args = parser.parse_args()

    game = resolve(args.rom, args.game)
    task = get_task(args.task, game.key)
    check_dataset(args.data, game.key, task.key)
    state = args.state or task.state
    horizon = args.horizon or task.horizon

    device = t.device(args.device)
    checkpoint = t.load(args.checkpoint, map_location=device, weights_only=False)
    model = JEPA(num_frames=checkpoint["config"]["num_frames"]).to(device)
    if args.untrained:
        # Keep the trained encoder so the goal embeddings are the same ones the
        # real planner sees; only the dynamics are random.
        model.load_state_dict(checkpoint["jepa"])
        fresh = JEPA(num_frames=checkpoint["config"]["num_frames"]).to(device)
        model.predictor.load_state_dict(fresh.predictor.state_dict())
        model.action_encoder.load_state_dict(fresh.action_encoder.state_dict())
    else:
        model.load_state_dict(checkpoint["jepa"])
    model.eval()

    start = start_index(args.data)
    goals_at = goal_indices(args.data, limit=args.max_goals)
    with t.inference_mode():
        initial = model.encode(load_frame(args.data, start).to(device))
        goals = t.cat(
            [model.encode(load_frame(args.data, index).to(device)) for index in goals_at]
        )

    env = PokeEnv(
        args.rom, state, headless=True, act_frames=args.act_frames, game=args.game
    )
    rows = []
    started = time.perf_counter()
    try:
        for offset in range(args.seeds):
            seed = args.first_seed + offset
            sequence, cost, _ = cem_plan(
                model, initial, goals,
                horizon=horizon, samples=args.samples, elites=args.elites,
                iterations=args.iterations, seed=seed, verbose=False,
            )
            indices = sequence.tolist()
            success, actions = run_episode(env, task, indices)
            rows.append({
                "seed": seed,
                "success": bool(success),
                "predicted_cost": cost,
                "actions_taken": actions,
                "plan": [ACTIONS[i] for i in indices],
            })
            hits = sum(r["success"] for r in rows)
            print(
                f"seed={seed:3d} success={str(success):5s} cost={cost:.6f} "
                f"actions={actions:3d}  running={hits}/{len(rows)}"
            )
    finally:
        env.close()

    elapsed = time.perf_counter() - started
    costs = np.array([r["predicted_cost"] for r in rows])
    successes = np.array([r["success"] for r in rows])
    auc = rank_auc(costs, successes)
    hits = int(successes.sum())

    print(f"\nsuccesses={hits}/{len(rows)} rate={hits / len(rows):.1%}")
    print(f"mean predicted cost  success {costs[successes].mean():.6f}"
          if hits else "mean predicted cost  success n/a")
    print(f"mean predicted cost  failure {costs[~successes].mean():.6f}"
          if hits < len(rows) else "mean predicted cost  failure n/a")
    if auc is None:
        print("cost->success AUC    n/a (one outcome only)")
    else:
        print(f"cost->success AUC    {auc:.3f}"
              f"{'   planner cannot tell its good plans from its bad ones' if abs(auc - 0.5) < 0.05 else ''}")
    print(f"wall clock {elapsed:.1f}s for {len(rows)} plans "
          f"({elapsed / len(rows):.2f}s each)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "game": game.key,
        "task": task.key,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "untrained_predictor": args.untrained,
        "horizon": horizon,
        "planner": {"samples": args.samples, "elites": args.elites,
                    "iterations": args.iterations, "device": str(device)},
        "goal_indices": list(goals_at),
        "successes": hits,
        "seeds": len(rows),
        "rate": hits / len(rows),
        "cost_success_auc": auc,
        "wall_clock_s": elapsed,
        "rows": rows,
    }, indent=2) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
