"""Build a behaviour-cloning dataset for Part 2 from the oracle's solution.

TRAINING ONLY: the oracle searches with emulator snapshots, which the
play-time contract bans. What comes out is a set of (frame, button) pairs,
and a policy trained on them still plays under the contract, reading pixels
and nothing else.

Why the oracle and not a gameplay video: a video carries the frames and not
the buttons, and the buttons are not recoverable from the pixels -- walking
into a wall and pressing nothing are the same picture. Why not the Part 1
planner either: its milestones are `pikachu, rival_beaten, left_lab,
left_pallet`, which is the opening. Part 2 is `get_parcel, deliver_parcel,
enter_forest` from Route 1, and the oracle is the only thing in this repo
that solves exactly that -- 487 actions, the measured ceiling.

The one subtlety is where each frame comes from. Replaying the oracle's
action list from the start state does not reproduce its route: the search
restores between actions and PyBoy's save/load is not bit-exact for whatever
that depends on, so the parcel phase's 87 actions end on the wrong tile.
Every search node stores the state it was reached in, so each edge is
restored to the state its button was chosen from and drift cannot accumulate
across edges. `restore` then `screen` is bit-identical, measured.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from poke.actions import ACTIONS, to_multihot
from poke.env import PokeEnv
from poke.oracle import nodes_for, search
from poke.ram import coords, cur_map, party_count
from poke.writer import EpisodeWriter

#: Which Part 2 objective each oracle phase is a demonstration of, so a clone
#: can be conditioned on the objective the way the RL policy is.
PHASE_OBJECTIVE = {
    "parcel": "get_parcel",
    "pokedex": "deliver_parcel",
    "forest": "enter_forest",
}


@dataclass
class DemoStep:
    frame: np.ndarray
    action_name: str
    location: tuple | None = None


def demo_steps(env, edge_blobs, edge_traces) -> list[DemoStep]:
    """Replay the solution edge by edge, capturing the frame before each press.

    Each edge is restored rather than continued, so an edge that does not
    reproduce exactly cannot push the next one off course.
    """
    steps: list[DemoStep] = []
    for blob, trace in zip(edge_blobs, edge_traces):
        env.restore(blob)
        for button in trace:
            frame = np.ascontiguousarray(env.screen()[:, :, 0])
            where = None
            try:
                where = (cur_map(env), *coords(env))
            except Exception:
                where = None
            steps.append(DemoStep(frame=frame, action_name=button, location=where))
            env.step(ACTIONS.index(button))
    return steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rom", default="pokeyellow.gbc")
    parser.add_argument("--state", default="states/oracle/route1.state")
    parser.add_argument("--out", default="episodes/demos_part2.h5")
    parser.add_argument("--report", default="results/demos_part2.json")
    parser.add_argument("--max-nodes", type=int, default=None)
    parser.add_argument("--only", nargs="+", default=None)
    args = parser.parse_args()

    from scripts.oracle import phases

    env = PokeEnv(args.rom, args.state)
    env.reset()
    writer = EpisodeWriter(args.out, args.rom, args.state, 60, 0, "yellow",
                           task="demo_part2")
    report = {"state": args.state, "phases": {}, "steps": 0,
              "what": "Oracle solution for Part 2 as (frame, button) pairs.",
              "contract": "training only: the search uses snapshots"}
    total = 0
    try:
        for name, goal, allowed, buttons in phases(env):
            if args.only and name not in args.only:
                continue
            print(f"searching {name} ...", flush=True)
            result = search(env, goal, allowed, buttons=buttons,
                            max_nodes=args.max_nodes or nodes_for())
            if not result.found:
                print(f"  {name}: NOT FOUND after {result.nodes} nodes", flush=True)
                report["phases"][name] = {"found": False, "nodes": result.nodes,
                                          "seconds": result.seconds, "steps": 0}
                break
            steps = demo_steps(env, result.edge_blobs, result.edge_traces)
            for step in steps:
                writer.add_step(step.frame, to_multihot(ACTIONS.index(step.action_name)),
                                party_count(env), False, step.location)
            writer.end_episode("demo", PHASE_OBJECTIVE[name], True, -1, -1)
            total += len(steps)
            report["phases"][name] = {
                "found": True, "edges": len(result.path), "steps": len(steps),
                "nodes": result.nodes, "seconds": result.seconds,
                "end": result.end, "objective": PHASE_OBJECTIVE[name],
            }
            print(f"  {name}: {len(result.path)} edges, {len(steps)} demo steps, "
                  f"{result.seconds}s, ends {result.end}", flush=True)
            # Leave the env on the solved state so the next phase starts there.
            env.restore(result.edge_blobs[-1])
            for button in result.edge_traces[-1]:
                env.step(ACTIONS.index(button))
    finally:
        writer.close()
        env.close()

    report["steps"] = total
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=1))
    print(json.dumps({"steps": total,
                      "phases": {k: v.get("steps") for k, v in report["phases"].items()}},
                     indent=1))


if __name__ == "__main__":
    main()
