"""Collect gameplay into an HDF5 dataset. Adapted from stmonty/lePokeRed.

Changes from upstream:
  * the task and its goal predicate come from `poke/tasks.py`, so success is
    not assumed to mean "party count went up in Oak's lab";
  * the demonstration route is loaded from disk and checked against the save
    state it was recorded on, instead of being a hard-coded `["a"] * 12`;
  * noisy variants are built by perturbing the recorded route (prefix noise
    and mid-route substitutions), which is what gives the predictor the bad
    action sequences a planner will later propose;
  * the per-step goal flag is written out, so the planner can pick goal frames
    for a task that leaves no trace in the party.
"""

from __future__ import annotations

import argparse

import numpy as np

from poke.actions import ACTIONS, to_multihot
from poke.env import PokeEnv
from poke.games import resolve
from poke.ram import party_count, party_species
from poke.routes import Route
from poke.tasks import get as get_task
from poke.writer import EpisodeWriter

NOOP = ACTIONS.index("noop")


def record(env, writer, task, action_idxs, kind, name):
    frame = env.reset()
    context = task.start_context(env)
    reached = task.goal(env, context)
    goal_step = 0 if reached else -1

    for step, idx in enumerate(action_idxs, start=1):
        writer.add_step(frame[..., 0], to_multihot(idx), party_count(env), reached)
        frame = env.step(idx)
        if not reached and task.goal(env, context):
            reached, goal_step = True, step

    writer.add_step(frame[..., 0], to_multihot(NOOP), party_count(env), reached)
    species = party_species(env)
    writer.end_episode(kind, name, reached, -1 if species is None else species, goal_step)
    return reached, species


def noisy_variants(route_indices, count, prefix_max, substitutions, rng):
    """Perturbed copies of the demonstration.

    Prefix noise makes the model see the route starting from a displaced
    position; substitutions corrupt the middle, so B presses and walls appear
    in contexts the clean route never visits.
    """
    variants = []
    n_actions = len(ACTIONS)
    for i in range(count):
        prefix = rng.integers(0, n_actions, size=int(rng.integers(0, prefix_max + 1))).tolist()
        body = list(route_indices)
        for _ in range(int(rng.integers(0, substitutions + 1))):
            if body:
                body[int(rng.integers(0, len(body)))] = int(rng.integers(0, n_actions))
        variants.append((f"noisy{i}", prefix + body))
    return variants


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rom", default="pokeyellow.gb")
    p.add_argument("--task", default="yellow_pikachu")
    p.add_argument("--state")
    p.add_argument("--route")
    p.add_argument("--out", default="episodes/poke.h5")
    p.add_argument("--demos", type=int, default=8, help="clean route repeats")
    p.add_argument("--noisy", type=int, default=200)
    p.add_argument("--noisy-prefix", type=int, default=3)
    p.add_argument("--noisy-substitutions", type=int, default=4)
    p.add_argument("--random", type=int, default=800)
    p.add_argument("--random-steps", type=int, default=48)
    p.add_argument("--act-frames", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--game")
    args = p.parse_args()

    game = resolve(args.rom, args.game)
    task = get_task(args.task, game.key)
    state = args.state or task.state
    route_path = args.route or task.route
    if route_path is None:
        raise SystemExit(
            f"task {task.key} has no demonstration route. Record one first:\n"
            f"  uv run python -m scripts.record_route --rom {args.rom} --task {task.key}"
        )

    route = Route.load(route_path)
    route.check(game.key, task.key, state)
    route_indices = route.indices

    rng = np.random.default_rng(args.seed)
    env = PokeEnv(args.rom, state, headless=True, act_frames=args.act_frames, game=args.game)

    print(f"game={game.key} task={task.key} state={state} route={route_path} "
          f"({len(route_indices)} actions, goal at {route.goal_step})")

    with EpisodeWriter(
        args.out, args.rom, state, args.act_frames, args.seed,
        game=game.key, task=task.key, route=route_path,
    ) as w:
        hits = 0
        for i in range(args.demos):
            ok, species = record(env, w, task, route_indices, "demo", f"demo{i}")
            hits += ok
            if not ok:
                raise SystemExit(
                    "the recorded route did not reach the goal on replay. The "
                    "emulator is deterministic from a save state, so this means "
                    "the route and the state disagree. Re-record the route."
                )
        print(f"demo     {args.demos} episodes  goal_hits={hits}  species={species}")

        hits = 0
        for name, plan in noisy_variants(
            route_indices, args.noisy, args.noisy_prefix, args.noisy_substitutions, rng
        ):
            ok, _ = record(env, w, task, plan, "noisy", name)
            hits += ok
        print(f"noisy    {args.noisy} episodes  goal_hits={hits}")

        hits = 0
        for i in range(args.random):
            idxs = rng.integers(0, len(ACTIONS), size=args.random_steps).tolist()
            ok, _ = record(env, w, task, idxs, "random", f"random{i}")
            hits += ok
            if (i + 1) % 100 == 0:
                print(f"random   {i + 1}/{args.random}")
        print(f"random   {args.random} episodes x {args.random_steps} steps  goal_hits={hits}")

    env.close()
    print("wrote", args.out)


if __name__ == "__main__":
    main()
