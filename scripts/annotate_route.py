"""Mark which steps of a route are walked and which are a cutscene.

Waypoints taken from a demonstration cannot tell "somewhere you walked to" from
"somewhere you were carried". That is not a detail: the pursuit planner reaches
(0, 10, 9) exactly, and nothing happens, because in the demonstration that tile
was recorded INSIDE Oak's encounter -- the game was moving the player. Two
presses of any button from the demonstration's state there enter the lab, but
walking to the same tile under your own power does not.

The test is a counterfactual the emulator can answer exactly. At each step,
snapshot, try all seven actions, and compare where each one lands. If every
button produces the same screen, nothing the planner chooses matters there:
that step is scripted. Otherwise it is navigable.

Costs seven emulator steps per route action -- seconds for a 454-action route.

  uv run python -m scripts.annotate_route --task yellow_pikachu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from poke.actions import ACTIONS
from poke.env import PokeEnv
from poke.games import resolve
from poke.ram import coords, cur_map
from poke.routes import Route
from poke.tasks import get as get_task

SCHEMA = 1


def classify(env) -> bool:
    """True if the player is in control: two buttons give two different screens.

    The same test bootstrap uses to decide the intro has handed over. Position
    alone cannot answer it -- a cutscene moves the sprite with no input, and a
    text box swallows every button while nothing moves.
    """
    blob = env.snapshot()
    screens = []
    try:
        for button in ("down", "right"):
            env.restore(blob)
            env.step(ACTIONS.index(button))
            screens.append(env.screen()[:, :, 0].tobytes())
    finally:
        env.restore(blob)
    return screens[0] != screens[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rom", default="pokeyellow.gbc")
    parser.add_argument("--task", default="yellow_pikachu")
    parser.add_argument("--state")
    parser.add_argument("--route")
    parser.add_argument("--out")
    parser.add_argument("--act-frames", type=int, default=60)
    parser.add_argument("--game")
    args = parser.parse_args()

    game = resolve(args.rom, args.game)
    task = get_task(args.task, game.key)
    state = args.state or task.state
    route_path = args.route or task.route
    route = Route.load(route_path)
    route.check(game.key, task.key, state)
    out = Path(args.out or route_path.replace(".json", ".annotations.json"))

    env = PokeEnv(args.rom, state, headless=True,
                  act_frames=args.act_frames, game=args.game)
    flags, places = [], []
    try:
        env.reset()
        for step, name in enumerate(route.actions):
            flags.append(bool(classify(env)))
            places.append([cur_map(env), *coords(env)])
            env.step(ACTIONS.index(name))
        flags.append(bool(classify(env)))
        places.append([cur_map(env), *coords(env)])
    finally:
        env.close()

    navigable = sum(flags)
    runs, start = [], 0
    for index in range(1, len(flags) + 1):
        if index == len(flags) or flags[index] != flags[start]:
            runs.append((flags[start], start, index - 1))
            start = index

    print(f"{len(flags)} steps: {navigable} navigable, "
          f"{len(flags) - navigable} scripted "
          f"({100 * (len(flags) - navigable) / len(flags):.0f}%)")
    print(f"{len(runs)} runs; scripted stretches of 3+ steps:")
    for is_nav, first, last in runs:
        if not is_nav and last - first >= 2:
            print(f"  steps {first:4d}-{last:4d} ({last - first + 1:3d}) "
                  f"at map {places[first][0]} {tuple(places[first][1:])}")

    out.write_text(json.dumps({
        "schema": SCHEMA,
        "game": game.key,
        "task": task.key,
        "state_sha1": route.state_sha1,
        "route_actions": len(route.actions),
        "navigable": flags,
        "places": places,
    }, indent=2) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
