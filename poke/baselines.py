"""Trivial policies, run in the emulator.

The blog post reports two controls — uniform random actions (0/100) and the
same search driven by an untrained predictor (1/100) — and then notes in
passing that pressing A repeatedly already works from its start state. It
never puts that row in the table. It is in fact row one of the upstream
training data: `collect.py` ships `TAKE = ["a"] * 12` as a scripted route and
asserts it reaches a starter.

A planner that does not beat every constant-button policy has not been shown
to do anything. These run in the emulator, cost seconds, and are reported
alongside the planner.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from poke.actions import ACTIONS, to_multihot
from poke.tasks import Task


@dataclass
class Result:
    name: str
    successes: int
    runs: int
    mean_actions: float

    @property
    def rate(self) -> float:
        return self.successes / self.runs if self.runs else 0.0

    def row(self) -> str:
        return (
            f"{self.name:<22} {self.successes:>4}/{self.runs:<4} "
            f"{self.rate:>7.1%}  {self.mean_actions:>6.1f}"
        )


HEADER = f"{'policy':<22} {'hits':>9} {'rate':>7}  {'steps':>6}"


def run_episode(env, task: Task, action_indices) -> tuple[bool, int]:
    """Execute actions until the goal fires or the sequence runs out."""
    env.reset()
    context = task.start_context(env)
    if task.goal(env, context):
        return True, 0
    for taken, action_index in enumerate(action_indices, start=1):
        env.step(action_index)
        if task.goal(env, context):
            return True, taken
    return False, len(action_indices)


def constant_policy(env, task: Task, action_name: str, horizon: int) -> Result:
    index = ACTIONS.index(action_name)
    success, actions = run_episode(env, task, [index] * horizon)
    # The emulator is deterministic from a fixed save state, so one run is the
    # whole answer for a fixed sequence. Reporting runs=1 keeps that explicit
    # instead of replaying an identical episode N times.
    return Result(f"always {action_name}", int(success), 1, float(actions))


def random_policy(env, task: Task, horizon: int, runs: int, seed: int) -> Result:
    rng = np.random.default_rng(seed)
    successes = 0
    total_actions = 0
    for _ in range(runs):
        sequence = rng.integers(0, len(ACTIONS), size=horizon).tolist()
        success, actions = run_episode(env, task, sequence)
        successes += int(success)
        total_actions += actions
    return Result("uniform random", successes, runs, total_actions / max(runs, 1))


def run_all(env, task: Task, horizon: int | None = None, runs: int = 100, seed: int = 0):
    """Every constant-button policy, plus uniform random over `runs` seeds."""
    horizon = task.horizon if horizon is None else horizon
    results = [constant_policy(env, task, name, horizon) for name in ACTIONS]
    results.append(random_policy(env, task, horizon, runs, seed))
    return results


def replay_as_dataset_route(action_indices):
    """Multi-hot encoding of a baseline sequence, for writing into an h5."""
    return np.stack([to_multihot(index) for index in action_indices])
