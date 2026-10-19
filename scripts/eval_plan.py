"""Execute a saved action plan in PyBoy. Adapted from stmonty/lePokeRed.

Changes: success is the task's goal predicate rather than a party-count read,
and `--runs` is gone. The emulator is deterministic from a save state and the
plan is fixed, so replaying it N times gives N identical episodes; upstream's
`--runs 10` measures nothing. To vary anything you have to vary the planner's
seed, which is what scripts/sweep_seeds.py does.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from poke.actions import ACTIONS
from poke.baselines import run_episode
from poke.env import PokeEnv
from poke.games import resolve
from poke.ram import party_count, party_species
from poke.tasks import get as get_task


def load_plan(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as file:
        plan = json.load(file)

    action_indices = plan.get("action_indices")
    if not isinstance(action_indices, list) or not action_indices:
        raise ValueError("plan must contain a non-empty action_indices list")
    if any(
        not isinstance(index, int) or not 0 <= index < len(ACTIONS)
        for index in action_indices
    ):
        raise ValueError("plan contains an invalid action index")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="plans/poke.json")
    parser.add_argument("--rom", default="pokeyellow.gb")
    parser.add_argument("--task")
    parser.add_argument("--state")
    parser.add_argument("--act-frames", type=int, default=60)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--game")
    args = parser.parse_args()

    plan = load_plan(args.plan)
    game = resolve(args.rom, args.game)
    task_key = args.task or plan.get("task")
    if not task_key:
        raise SystemExit("plan has no task key; pass --task")
    task = get_task(task_key, game.key)
    if plan.get("game") and plan["game"] != game.key:
        raise SystemExit(f"plan was made for {plan['game']!r}, not {game.key!r}")
    state = args.state or task.state

    action_indices = plan["action_indices"]
    print(f"game={game.key} task={task.key}")
    print(f"plan: {' '.join(ACTIONS[index] for index in action_indices)}")

    env = PokeEnv(
        args.rom, state, headless=not args.show,
        act_frames=args.act_frames, game=args.game,
    )
    try:
        success, actions = run_episode(env, task, action_indices)
        print(
            f"success={success} actions={actions} "
            f"party={party_count(env)} species={party_species(env)}"
        )
        if "predicted_cost" in plan:
            print(f"predicted_cost={plan['predicted_cost']:.6f}")
    finally:
        env.close()

    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
