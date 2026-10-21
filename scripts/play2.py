"""Play Part 2 under the contract: one continuous run, no rollback.

The policy argument is deliberately dumb here -- random, a constant button, or
a replay of the oracle's path. The trained policy lands in this same slot, so
whatever this file measures about the baselines it will measure about it.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from poke.actions import ACTIONS, n_actions
from poke.contract import PlayTimeEnv, Run
from poke.env import PokeEnv
from poke.handlers import HandlerState, choose
from poke.objectives import CHAIN, one_hot

POLICIES = ("random", "noop", "up", "down", "left", "right", "a", "b", "oracle")


def policy_for(name: str, rng: random.Random, path=None):
    """A callable (env, objective one-hot) -> action index."""
    if name == "random":
        return lambda env, objective: rng.randrange(n_actions())
    if name == "oracle":
        steps = list(path or [])

        def replay(env, objective, _steps=steps):
            return ACTIONS.index(_steps.pop(0)) if _steps else ACTIONS.index("noop")

        return replay
    index = ACTIONS.index(name)
    return lambda env, objective: index


def play(env, policy, budget: int, run: Run | None = None) -> dict:
    """One episode. Handlers take the turns they own; the policy takes the rest.

    Handler actions count against the budget, as the spec requires: a run that
    spends its whole budget in menus has not navigated anywhere.
    """
    run = run or Run(env)
    guarded = PlayTimeEnv(env)
    state = HandlerState()
    # An objective that is already true when the run starts is not something
    # the run did. Stage A starts on Route 1, where trigger_oak and
    # get_starter are already flagged, and crediting those made every constant
    # button look like it had cleared two objectives.
    already = {objective.key for objective in CHAIN if objective.reached(env)}
    reached: dict[str, int | None] = {objective.key: None for objective in CHAIN}

    for _ in range(budget):
        before_map = env.read_ram(env.game.ram.cur_map)
        claimed = choose(env, state)
        if claimed is None:
            action, controller = policy(guarded, one_hot(env)), "policy"
        else:
            action, controller = claimed
        env.step(action)
        run.record(action, controller)
        state.note(action, env.read_ram(env.game.ram.cur_map) != before_map)
        for objective in CHAIN:
            if (objective.key not in already
                    and reached[objective.key] is None
                    and objective.reached(env)):
                reached[objective.key] = len(run.log)

    return {
        "objectives": reached,
        "already_true_at_start": sorted(already),
        "actions": len(run.log),
        "by_controller": dict(run.by_controller),
        "learned_share": run.learned_share,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rom", default="pokeyellow.gbc")
    parser.add_argument("--state", default="states/oracle/route1.state")
    parser.add_argument("--policy", default="random", choices=POLICIES)
    parser.add_argument("--oracle-report", default="results/oracle.json",
                        help="where --policy oracle reads its path from")
    parser.add_argument("--budget", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    path = None
    if args.policy == "oracle":
        report = json.loads(Path(args.oracle_report).read_text())
        path = [button for phase in report.values() for button in phase.get("path", [])]

    env = PokeEnv(args.rom, args.state)
    env.reset()
    env.settle()
    result = play(env, policy_for(args.policy, random.Random(args.seed), path),
                  budget=args.budget)
    result["policy"] = args.policy
    result["seed"] = args.seed
    result["state"] = args.state
    print(json.dumps(result, indent=1))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
