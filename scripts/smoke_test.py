"""One reset plus one A press. From stmonty/lePokeRed.

Added: prints the detected game and its RAM readings, which is the fastest way
to catch a save state made in the wrong place or a RAM map that does not fit
this ROM.
"""

import argparse

from PIL import Image

from poke.actions import ACTIONS
from poke.env import PokeEnv
from poke.ram import cur_map, party_count, party_species
from poke.tasks import get as get_task


def save(frame, path, scale=3):
    img = Image.fromarray(frame)
    img = img.resize((frame.shape[1] * scale, frame.shape[0] * scale), Image.NEAREST)
    img.save(path)
    print("wrote", path, "shape", frame.shape)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rom", default="pokeyellow.gb")
    p.add_argument("--task", default="yellow_pikachu")
    p.add_argument("--state")
    p.add_argument("--game")
    args = p.parse_args()

    env = PokeEnv(args.rom, args.state or get_task(args.task).state,
                  headless=True, game=args.game)
    try:
        task = get_task(args.task, env.game.key)
        before = env.reset()
        context = task.start_context(env)
        save(before, "before.png")
        print(f"game={env.game.key} task={task.key}")
        print(f"  party_count@{env.game.ram.party_count:#06x} = {party_count(env)}")
        print(f"  party_species@{env.game.ram.party_species:#06x} = {party_species(env)}")
        print(f"  cur_map@{env.game.ram.cur_map:#06x} = {cur_map(env)} "
              f"({env.game.ram.confidence.get('cur_map')})")
        print(f"  goal already satisfied: {task.goal(env, context)}")

        after = env.step(ACTIONS.index("a"))
        save(after, "after.png")
        print(f"  after one A: party_count = {party_count(env)}, "
              f"goal = {task.goal(env, context)}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
