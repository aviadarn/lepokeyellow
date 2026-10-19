"""The baseline table the blog post is missing.

Run this before believing any planner number. If `always a` solves the task,
the task does not measure planning -- which is the case for the upstream Red
start state, where the shipped scripted route is literally twelve A presses.

  uv run python -m scripts.eval_baselines --rom pokeyellow.gb --task yellow_pikachu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from poke.baselines import HEADER, run_all
from poke.env import PokeEnv
from poke.games import resolve
from poke.tasks import get as get_task


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rom", default="pokeyellow.gb")
    p.add_argument("--task", default="yellow_pikachu")
    p.add_argument("--state")
    p.add_argument("--horizon", type=int)
    p.add_argument("--runs", type=int, default=100, help="seeds for uniform random")
    p.add_argument("--act-frames", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--game")
    p.add_argument("--out", default="results/baselines.json")
    args = p.parse_args()

    game = resolve(args.rom, args.game)
    task = get_task(args.task, game.key)
    state = args.state or task.state
    horizon = args.horizon or task.horizon

    env = PokeEnv(args.rom, state, headless=True, act_frames=args.act_frames, game=args.game)
    try:
        results = run_all(env, task, horizon=horizon, runs=args.runs, seed=args.seed)
    finally:
        env.close()

    print(f"game={game.key} task={task.key} horizon={horizon}")
    print(task.description)
    print()
    print(HEADER)
    for result in results:
        print(result.row())

    solved_by_constant = [r.name for r in results if r.runs == 1 and r.successes]
    print()
    if solved_by_constant:
        print(
            "WARNING: solved by a single repeated button "
            f"({', '.join(solved_by_constant)}). A planner beating random on "
            "this task has not been shown to plan."
        )
    else:
        print("No constant-button policy solves this task. A planner has something to do.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "game": game.key,
        "task": task.key,
        "horizon": horizon,
        "random_runs": args.runs,
        "seed": args.seed,
        "results": [
            {"name": r.name, "successes": r.successes, "runs": r.runs,
             "rate": r.rate, "mean_actions": r.mean_actions}
            for r in results
        ],
        "solved_by_constant_button": solved_by_constant,
    }, indent=2) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
