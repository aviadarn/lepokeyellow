"""Play in a window; closing it saves a save state. From stmonty/lePokeRed.

Added: the ROM is identified first, so a Yellow state is not written under a
Red task's filename, and the party count is printed on exit -- the Yellow task
is defined from an *empty* party, and a state saved one scene too late is the
easiest way to produce a task that is already solved at step 0.
"""

from pathlib import Path

from pyboy import PyBoy

from poke.games import resolve
from poke.utils import default_parser


def main() -> None:
    parser = default_parser(rom=True, state=False, out=None)
    parser.add_argument("--out")
    parser.add_argument("--game")
    args = parser.parse_args()

    game = resolve(args.rom, args.game)
    out = args.out or f"states/{game.key}_lab.state"
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    print(f"game: {game.title} ({game.key})")
    if game.key == "yellow":
        print("Stop inside Oak's lab with an EMPTY party, before taking the ball.")
    else:
        print("Stop facing the middle ball in Oak's lab, no dialogue, empty party.")
    print("Close the window to save.")

    pyboy = PyBoy(args.rom, window="SDL2")
    pyboy.set_emulation_speed(target_speed=1)
    while pyboy.tick():
        pass

    count = pyboy.memory[game.ram.party_count]
    with open(out, "wb") as handle:
        pyboy.save_state(handle)
    pyboy.stop()

    print(f"wrote {out}  party_count@{game.ram.party_count:#06x} = {count}")
    if count:
        print("WARNING: party is not empty, so a 'get a Pokemon' task starts solved.")


if __name__ == "__main__":
    main()
