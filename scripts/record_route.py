"""Record one demonstration route by hand, in the env's own action space.

The agent acts once per `act_frames` (60 by default: button held 20, then 40
to settle). Playing in real time would not line up with that, so this is a
stepper: the emulator window shows the game, you press one key in the
terminal, one action executes, repeat. What you record is exactly what the
model and the planner can express.

  w a s d  up left down right      j  A      k  B
  .        noop                    u  undo last action
  q        save and quit           Q  quit without saving

Undo replays the route from the save state, so it costs one emulator pass.
"""

from __future__ import annotations

import argparse
import sys
import termios
import tty

from poke.actions import ACTIONS
from poke.env import PokeEnv
from poke.games import resolve
from poke.ram import party_count
from poke.routes import SCHEMA, Route, state_digest
from poke.tasks import get as get_task

KEYS = {
    "w": "up", "s": "down", "a": "left", "d": "right",
    "j": "a", "k": "b", ".": "noop",
}


def read_key() -> str:
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def replay(env, task, action_names):
    """Reset and re-execute `action_names`; return (context, reached, step)."""
    env.reset()
    context = task.start_context(env)
    reached, goal_step = task.goal(env, context), -1
    if reached:
        goal_step = 0
    for step, name in enumerate(action_names, start=1):
        env.step(ACTIONS.index(name))
        if not reached and task.goal(env, context):
            reached, goal_step = True, step
    return context, reached, goal_step


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rom", default="pokeyellow.gb")
    parser.add_argument("--task", default="yellow_pikachu")
    parser.add_argument("--state")
    parser.add_argument("--out")
    parser.add_argument("--act-frames", type=int, default=60)
    parser.add_argument("--game")
    args = parser.parse_args()

    game = resolve(args.rom, args.game)
    task = get_task(args.task, game.key)
    state = args.state or task.state
    out = args.out or task.route
    if out is None:
        raise SystemExit(f"task {task.key} defines no route file; pass --out")

    # Not headless: the window is how you see what you are doing.
    env = PokeEnv(args.rom, state, headless=False, act_frames=args.act_frames, game=args.game)
    recorded: list[str] = []
    context, reached, goal_step = replay(env, task, recorded)

    print(f"game {game.key}  task {task.key}  state {state}")
    print(f"party_count@{game.ram.party_count:#06x} = {party_count(env)}")
    print(__doc__.split("\n\n")[-1])

    try:
        while True:
            status = "GOAL" if reached else "...."
            print(
                f"\r[{status}] {len(recorded):3d} actions  party={party_count(env)}  > ",
                end="",
                flush=True,
            )
            key = read_key()
            if key == "Q":
                print("\nquit without saving")
                return
            if key == "q":
                break
            if key == "u":
                if recorded:
                    recorded.pop()
                    context, reached, goal_step = replay(env, task, recorded)
                continue
            if key not in KEYS:
                continue
            name = KEYS[key]
            recorded.append(name)
            env.step(ACTIONS.index(name))
            if not reached and task.goal(env, context):
                reached, goal_step = True, len(recorded)
    finally:
        env.close()

    if not recorded:
        raise SystemExit("nothing recorded")

    route = Route(
        schema=SCHEMA,
        game=game.key,
        task=task.key,
        state_sha1=state_digest(state),
        actions=recorded,
        reached_goal=reached,
        goal_step=goal_step,
    )
    route.save(out)
    print(f"\nwrote {out}: {len(recorded)} actions, reached_goal={reached}, goal_step={goal_step}")
    if not reached:
        print("WARNING: route never reached the goal. collect.py will refuse it.")


if __name__ == "__main__":
    main()
