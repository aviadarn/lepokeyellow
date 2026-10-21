"""Play Part 2 under the contract: one continuous run, no rollback.

The policy argument is deliberately dumb here -- random, or a constant button.
The trained policy lands in this same slot, so whatever this file measures
about the baselines it will measure about it.

The oracle is NOT one of these. Its search snapshots and restores between
actions, and the resulting action log does not replay from a clean load:
measured, its 87-action first phase ends at (12, 28, 9) on Route 1 instead of
at the Viridian Mart. It is a ceiling expressed as a number -- see
results/part2_ceiling.json -- not a run.
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

POLICIES = ("random", "noop", "up", "down", "left", "right", "a", "b")


def policy_for(name: str, rng: random.Random):
    """A callable (env, objective one-hot) -> action index."""
    if name == "random":
        return lambda env, objective: rng.randrange(n_actions())
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
    parser.add_argument("--budget", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    env = PokeEnv(args.rom, args.state)
    env.reset()
    # No settle(). It ticks frames without an action, and both the oracle's
    # recorded path and any open-loop replay are frame-phase sensitive: with
    # it, the same buttons walk a different route. reset() already renders one
    # frame, which is all the screen detector needs.
    result = play(env, policy_for(args.policy, random.Random(args.seed)),
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
