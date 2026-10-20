"""Cut the loops out of a route, then prove the short version still works.

`scripts/bootstrap.py` finds a route by random walk, which is fine for what it
is for -- an existence proof that produces a reproducible save state. It is a
terrible demonstration. The route to Pikachu is 1,485 actions of flailing, and
the subgoal ladder built from its frames is a random tour rather than a
curriculum: a play run climbed 7 rungs of it without ever leaving the bedroom.

A random walk that returns to a tile it has already stood on wasted every
action in between. So: replay the route, record a signature per step, and
whenever a signature repeats, delete the actions between the two visits. Repeat
until nothing shrinks.

The signature is (map, y, x, party, battle), which is not the whole game state
-- story flags are not in it -- so a cut can skip a scripted trigger and break
the route. Every candidate is therefore REPLAYED and checked against the goal
before it is accepted, and a cut that breaks the route is rolled back. The
output is a route that is both shorter and verified, or the input unchanged.

  uv run python -m scripts.prune_route --task yellow_pikachu
"""

from __future__ import annotations

import argparse
import functools
import sys
import time
from pathlib import Path

print = functools.partial(print, flush=True)  # noqa: A001 - progress must stream

from poke.actions import ACTIONS
from poke.env import PokeEnv
from poke.games import resolve
from poke.ram import battle_kind, coords, cur_map, party_count
from poke.routes import SCHEMA, Route, state_digest
from poke.tasks import get as get_task


def signature(env) -> tuple:
    y, x = coords(env)
    return (cur_map(env), y, x, party_count(env), battle_kind(env))


def replay(env, task, actions: list[str]) -> tuple[bool, int, list[tuple]]:
    """Run `actions` from the start state. Returns (reached, step, signatures)."""
    env.reset()
    context = task.start_context(env)
    signatures = [signature(env)]
    reached, goal_step = task.goal(env, context), -1
    if reached:
        goal_step = 0
    for step, name in enumerate(actions, start=1):
        env.step(ACTIONS.index(name))
        signatures.append(signature(env))
        if not reached and task.goal(env, context):
            reached, goal_step = True, step
            break
    return reached, goal_step, signatures


def shortcut_all(signatures: list[tuple]) -> list[int] | None:
    """Indices of the actions to keep after removing every loop in one pass.

    Walks the signature list and, at each position, jumps straight to the LAST
    step with the same signature. One pass removes every cycle at once, so the
    expensive part -- replaying to verify -- happens twice instead of once per
    cut. Returns None when there is nothing to remove.

    Action i takes the game from signatures[i] to signatures[i + 1], so keeping
    the hop from step `a` to step `b` means keeping action index `a`.
    """
    last_seen: dict[tuple, int] = {}
    for index, sig in enumerate(signatures):
        last_seen[sig] = index

    keep: list[int] = []
    position = 0
    while position < len(signatures) - 1:
        jump = last_seen[signatures[position]]
        if jump > position:
            position = jump
            continue
        keep.append(position)
        position += 1
    return keep if len(keep) < len(signatures) - 1 else None


def find_loop(signatures: list[tuple]) -> tuple[int, int] | None:
    """First repeated signature, as the (earliest, latest) step pair.

    The fallback when a whole-pass shortcut breaks the route: cuts one loop at
    a time so a single bad cut does not discard the good ones. Taking the
    widest span for the earliest repeat removes the most actions per
    verification.
    """
    for index in range(len(signatures)):
        sig = signatures[index]
        latest = max(i for i, s in enumerate(signatures) if s == sig)
        if latest > index:
            return (index, latest)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rom", default="pokeyellow.gbc")
    parser.add_argument("--task", default="yellow_pikachu")
    parser.add_argument("--state")
    parser.add_argument("--route")
    parser.add_argument("--out")
    parser.add_argument("--max-cuts", type=int, default=400)
    parser.add_argument("--act-frames", type=int, default=60)
    parser.add_argument("--game")
    args = parser.parse_args()

    game = resolve(args.rom, args.game)
    task = get_task(args.task, game.key)
    state = args.state or task.state
    route_path = args.route or task.route
    route = Route.load(route_path)
    route.check(game.key, task.key, state)
    out = Path(args.out or route_path)

    env = PokeEnv(args.rom, state, headless=True,
                  act_frames=args.act_frames, game=args.game)
    started = time.perf_counter()
    try:
        actions = list(route.actions)
        reached, goal_step, signatures = replay(env, task, actions)
        if not reached:
            raise SystemExit(f"{route_path} does not reach the goal on replay")
        actions = actions[:goal_step]
        signatures = signatures[: goal_step + 1]
        print(f"start: {len(actions)} actions, goal at {goal_step}")

        # One pass first: remove every loop at once and verify. When it holds
        # -- the common case, since most of a random walk is cycles -- the job
        # is done in two replays instead of one per cut.
        cuts, rejected = 0, 0
        keep = shortcut_all(signatures)
        if keep is not None:
            candidate = [actions[i] for i in keep]
            print(f"  one-pass shortcut: {len(actions)} -> {len(candidate)} actions")
            ok, step, new_signatures = replay(env, task, candidate)
            if ok:
                actions = candidate[:step]
                signatures = new_signatures[: step + 1]
                cuts += 1
                print(f"  verified: {len(actions)} actions reach the goal")
            else:
                print("  one-pass shortcut broke the route (a cut skipped a "
                      "scripted trigger); falling back to one loop at a time")

        while cuts < args.max_cuts:
            loop = find_loop(signatures)
            if loop is None:
                break
            start, end = loop
            candidate = actions[:start] + actions[end:]
            if len(candidate) >= len(actions):
                break
            ok, step, new_signatures = replay(env, task, candidate)
            if ok:
                removed = len(actions) - len(candidate)
                actions = candidate[:step]
                signatures = new_signatures[: step + 1]
                cuts += 1
                print(f"  cut {cuts:3d}: -{removed:4d} actions "
                      f"(steps {start}-{end}) -> {len(actions)}")
            else:
                # The signature misses story flags, so this cut skipped a
                # trigger. Break the loop record rather than the route.
                rejected += 1
                signatures[end] = signatures[end] + ("kept",)
                if rejected > args.max_cuts:
                    break

        ok, goal_step, _ = replay(env, task, actions)
        if not ok:
            raise SystemExit("final route does not reach the goal; refusing to write")
    finally:
        env.close()

    Route(
        schema=SCHEMA,
        game=game.key,
        task=task.key,
        state_sha1=state_digest(state),
        actions=actions,
        reached_goal=True,
        goal_step=goal_step,
    ).save(out)

    elapsed = time.perf_counter() - started
    print(f"\n{len(route.actions)} -> {len(actions)} actions "
          f"({100 * (1 - len(actions) / len(route.actions)):.1f}% shorter), "
          f"{cuts} cuts accepted, {rejected} rejected")
    print(f"wrote {out}  ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
