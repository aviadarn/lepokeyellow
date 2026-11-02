"""Run the oracle over Part 2 and write what training needs from it.

TRAINING ONLY: this uses emulator snapshots, which the play-time contract bans.

Writes one save state per phase into --states-dir and a JSON report with the
path, the node counts and the wall-clock cost of each phase.

With --color it runs in GBC mode from states/cgb/, starting at the bedroom and
running Part 1 (pikachu, rival, route1) first. The DMG Route 1 state came from
replaying a play log, which is frame-phase sensitive and does not transfer to
CGB timing, so in colour the search walks Part 1 too.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from poke.actions import ACTIONS
from poke.env import PokeEnv
from poke.flags import BATTLED_RIVAL_IN_OAKS_LAB, GOT_POKEDEX, carrying_parcel, event_bit
from poke.objectives import VIRIDIAN_FOREST
from poke.oracle import MAX_SEARCH_BYTES, nodes_for, search
from poke.ram import has_pokemon, in_battle

FOREST_GATE = 50
ROUTE_1 = 12
PART1 = ("pikachu", "rival", "route1")


def phases(env):
    """(name, goal, allowed maps, buttons) in the order the story needs."""
    return [
        # Bedroom -> Oak stops the player north of Pallet -> lab -> Pikachu.
        ("pikachu", lambda: has_pokemon(env), {38, 37, 0, 40},
         ("up", "down", "left", "right", "a")),
        # The rival challenges the player on the way out of the lab.
        ("rival", lambda: (event_bit(env, BATTLED_RIVAL_IN_OAKS_LAB) == 1
                           and not in_battle(env)), {40},
         ("up", "down", "left", "right", "a")),
        ("route1", lambda: env.read_ram(env.game.ram.cur_map) == ROUTE_1,
         {40, 0, ROUTE_1}, ("up", "down", "left", "right")),
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
    parser.add_argument("--state", default=None,
                        help="where to start; default states/route1.state, or "
                             "states/cgb/bedroom.state with --color")
    parser.add_argument("--states-dir", default=None,
                        help="default states/oracle, or states/cgb/oracle")
    parser.add_argument("--out", default=None)
    parser.add_argument("--color", action="store_true",
                        help="GBC colour, Part 1 included")
    parser.add_argument("--max-nodes", type=int, default=None,
                        help="default: whatever fits in "
                             f"{MAX_SEARCH_BYTES // 1024 ** 3} GiB of save "
                             f"states ({nodes_for()} nodes)")
    parser.add_argument("--only", nargs="+", default=None,
                        help="run only these phases, e.g. --only forest")
    parser.add_argument("--replay", default=None,
                        help="a play.py result JSON to replay from --state "
                             "first; this is how the Route 1 start is reached "
                             "from a new game without shipping a save state")
    args = parser.parse_args()
    if args.state is None:
        args.state = ("states/cgb/bedroom.state" if args.color
                      else "states/route1.state")
    if args.states_dir is None:
        args.states_dir = "states/cgb/oracle" if args.color else "states/oracle"
    if args.out is None:
        args.out = "results/oracle_cgb.json" if args.color else "results/oracle.json"
    if args.only is None and not args.color:
        # DMG starts on Route 1, where Part 1 is already done.
        args.only = [name for name, *_ in phases(None) if name not in PART1]

    env = PokeEnv(args.rom, args.state, color=args.color)
    env.reset()
    if not args.replay:
        # Never before a replay. settle() ticks frames without an action, and
        # an open-loop log is frame-phase sensitive: with it, `down` at the
        # top of the stairs walks two tiles instead of one and the run ends up
        # in Oak's lab instead of on Route 1. Measured, twice now.
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
        start = Path(args.states_dir) / "route1.state"
        start.write_bytes(env.snapshot())
        print(f"wrote {start}", flush=True)
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
            "actions": result.actions, "action_len": len(result.actions),
            "path_len": len(result.path), "expanded": result.expanded,
            "nodes": result.nodes, "emulated_actions": result.emulated_actions,
            "maps_seen": result.maps_seen, "seconds": result.seconds,
            "end": result.end,
        }
        print(json.dumps({k: v for k, v in report[name].items()
                          if k not in ("path", "actions")}), flush=True)
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
