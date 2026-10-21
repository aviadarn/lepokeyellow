"""Run the oracle over Part 2 and write what training needs from it.

TRAINING ONLY: this uses emulator snapshots, which the play-time contract bans.

Writes one save state per phase into --states-dir and a JSON report with the
path, the node counts and the wall-clock cost of each phase.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from poke.actions import ACTIONS
from poke.env import PokeEnv
from poke.flags import GOT_POKEDEX, carrying_parcel, event_bit
from poke.objectives import VIRIDIAN_FOREST
from poke.oracle import search

FOREST_GATE = 50


def phases(env):
    """(name, goal, allowed maps, buttons) in the order the story needs."""
    return [
        ("parcel", lambda: carrying_parcel(env), {12, 1, 42},
         ("up", "down", "left", "right", "a")),
        ("pokedex", lambda: event_bit(env, GOT_POKEDEX) == 1, {42, 1, 12, 0, 40},
         ("up", "down", "left", "right", "a")),
        ("forest", lambda: env.read_ram(env.game.ram.cur_map) == VIRIDIAN_FOREST,
         {40, 0, 12, 1, 13, FOREST_GATE, VIRIDIAN_FOREST},
         ("up", "down", "left", "right")),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rom", default="pokeyellow.gbc")
    parser.add_argument("--state", default="states/route1.state",
                        help="where to start; the Route 1 state for stage A")
    parser.add_argument("--states-dir", default="states/oracle")
    parser.add_argument("--out", default="results/oracle.json")
    parser.add_argument("--max-nodes", type=int, default=20000)
    parser.add_argument("--only", nargs="+", default=None,
                        help="run only these phases, e.g. --only forest")
    parser.add_argument("--replay", default=None,
                        help="a play.py result JSON to replay from --state "
                             "first; this is how the Route 1 start is reached "
                             "from a new game without shipping a save state")
    args = parser.parse_args()

    env = PokeEnv(args.rom, args.state)
    env.reset()
    env.settle()
    if args.replay:
        log = json.loads(Path(args.replay).read_text())["log"]
        for entry in log:
            env.step(ACTIONS.index(entry["action"]))
        here = (env.read_ram(env.game.ram.cur_map),
                env.read_ram(env.game.ram.y_coord),
                env.read_ram(env.game.ram.x_coord))
        expected = (log[-1]["map"], log[-1]["y"], log[-1]["x"])
        if here != expected:
            raise SystemExit(
                f"replay diverged: at {here}, the log ends at {expected}. "
                "Open-loop replay is frame-phase sensitive; re-record."
            )
        print(f"replayed {len(log)} actions to {here}", flush=True)
    Path(args.states_dir).mkdir(parents=True, exist_ok=True)
    report = {}

    def progress(expanded, nodes, queued, maps):
        print(f"expanded {expanded} nodes {nodes} queue {queued} maps {maps}",
              flush=True)

    for name, goal, allowed, buttons in phases(env):
        if args.only and name not in args.only:
            continue
        result = search(env, goal, allowed, buttons=buttons,
                        with_facing="a" in buttons, max_nodes=args.max_nodes,
                        on_progress=progress)
        report[name] = {
            "found": result.found, "path": result.path,
            "path_len": len(result.path), "expanded": result.expanded,
            "nodes": result.nodes, "emulated_actions": result.emulated_actions,
            "maps_seen": result.maps_seen, "seconds": result.seconds,
            "end": result.end,
        }
        print(json.dumps({k: v for k, v in report[name].items() if k != "path"}),
              flush=True)
        if not result.found:
            break
        path = Path(args.states_dir) / f"{name}.state"
        path.write_bytes(env.snapshot())
        report[name]["state"] = str(path)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
