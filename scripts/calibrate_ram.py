"""Confirm this ROM's RAM addresses instead of trusting the table.

Yellow's WRAM sits one byte below Red/Blue's above $CF1A, and a wrong address
does not raise: it returns a neighbouring byte. `poke/games.py` marks Yellow's
`cur_map` as inferred from that shift rule rather than read off a disassembly,
so it needs confirming on a real cartridge dump before any result rests on it.

This diffs WRAM across a demonstration route and reports every byte that moved
the way the goal did. For a party-count task the answer is the byte that goes
0 -> 1 exactly once and never above 6.

  uv run python -m scripts.calibrate_ram --rom pokeyellow.gb --task yellow_pikachu
"""

from __future__ import annotations

import argparse

from poke.actions import ACTIONS
from poke.env import PokeEnv
from poke.games import resolve
from poke.routes import Route

WRAM_START, WRAM_STOP = 0xC000, 0xE000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rom", default="pokeyellow.gb")
    parser.add_argument("--task", default="yellow_pikachu")
    parser.add_argument("--state")
    parser.add_argument("--route")
    parser.add_argument("--act-frames", type=int, default=60)
    parser.add_argument("--game")
    parser.add_argument("--max-value", type=int, default=6)
    args = parser.parse_args()

    from poke.tasks import get as get_task

    game = resolve(args.rom, args.game)
    task = get_task(args.task, game.key)
    state = args.state or task.state
    route_path = args.route or task.route
    if route_path is None:
        raise SystemExit(
            f"task {task.key} has no route; record one with scripts/record_route.py"
        )
    route = Route.load(route_path)
    route.check(game.key, task.key, state)

    env = PokeEnv(args.rom, state, headless=True, act_frames=args.act_frames, game=args.game)
    try:
        env.reset()
        history = [env.read_ram_range(WRAM_START, WRAM_STOP)]
        for name in route.actions:
            env.step(ACTIONS.index(name))
            history.append(env.read_ram_range(WRAM_START, WRAM_STOP))
    finally:
        env.close()

    first, last = history[0], history[-1]
    candidates = []
    for offset in range(len(first)):
        if first[offset] != 0 or last[offset] != 1:
            continue
        series = [snapshot[offset] for snapshot in history]
        if max(series) > args.max_value:
            continue
        # A counter rises once and stays up; a flicker is something else.
        transitions = sum(
            1 for a, b in zip(series, series[1:]) if (a == 0) != (b == 0)
        )
        if transitions == 1:
            candidates.append((WRAM_START + offset, series.index(1)))

    print(f"game={game.key} route={route_path} steps={len(route.actions)}")
    print(f"table says party_count = {game.ram.party_count:#06x} "
          f"({game.ram.confidence.get('party_count', 'unknown')})")
    print(f"table says cur_map     = {game.ram.cur_map:#06x} "
          f"({game.ram.confidence.get('cur_map', 'unknown')})")
    print(f"\n{len(candidates)} byte(s) went 0 -> 1 once and stayed <= {args.max_value}:")
    for address, step in candidates:
        mark = "  <-- matches table" if address == game.ram.party_count else ""
        print(f"  {address:#06x}  first set at step {step}{mark}")

    if not any(address == game.ram.party_count for address, _ in candidates):
        print(
            f"\nMISMATCH: {game.ram.party_count:#06x} is not among the candidates. "
            "Do not trust party_count for this ROM until this is resolved."
        )
        raise SystemExit(1)
    print("\nparty_count confirmed for this ROM.")


if __name__ == "__main__":
    main()
