"""Record frames over the Part 2 maps, for the encoder refresh.

The oracle's action log does not replay -- measured, see
results/part2_ceiling.json -- so the demonstrations it could have provided are
not available. What it does provide is five save states, one per milestone, and
those are where the interesting maps are. So: start from each of them and walk.

Two walk kinds, because uniform random is a poor photographer. "random" is
uniform over the seven actions. "sticky" repeats the previous action with
probability STICKY, which is how you get down a corridor instead of vibrating
in a doorway -- the same trick the noisy variants in scripts/collect.py use.
Handlers run in both, so dialogue is advanced rather than stared at, and the
frames that land in the dataset are the frames a policy will actually see.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

from poke.actions import ACTIONS, n_actions, to_multihot
from poke.env import PokeEnv
from poke.handlers import HandlerState, choose
from poke.objectives import CHAIN
from poke.ram import coords, cur_map, party_count, party_species
from poke.writer import EpisodeWriter

#: One per milestone the oracle reached, plus the Part 1 bedroom state, so the
#: refreshed encoder keeps the maps the old one already knew.
MILESTONE_STATES = (
    "states/bedroom.state",
    "states/oracle/route1.state",
    "states/oracle/parcel.state",
    "states/oracle/pokedex.state",
    "states/oracle/forest_gate.state",
    "states/oracle/forest.state",
)

#: Probability a sticky walk repeats its last button.
STICKY = 0.7


@dataclass(frozen=True)
class Job:
    state: str
    seed: int
    kind: str


def episode_plan(states, per_state: int, seed: int) -> list[Job]:
    """Which episodes to record, in order. Pure: the tests run without a ROM."""
    jobs = []
    for index, state in enumerate(states):
        for episode in range(per_state):
            jobs.append(Job(
                state=state,
                seed=seed + index * per_state + episode,
                kind="random" if episode % 2 == 0 else "sticky",
            ))
    return jobs


def walk(rng: random.Random, kind: str, previous: int) -> int:
    if kind == "sticky" and previous >= 0 and rng.random() < STICKY:
        return previous
    return rng.randrange(n_actions())


def record_episode(env, writer, actions: int, seed: int, kind: str, name: str) -> dict:
    """One episode: walk, let the handlers work, write every frame."""
    frame = env.reset()
    rng = random.Random(seed)
    handlers = HandlerState()
    reached = {objective.key for objective in CHAIN if objective.reached(env)}
    previous, maps = -1, set()

    for _ in range(actions):
        claimed = choose(env, handlers)
        action = claimed[0] if claimed else walk(rng, kind, previous)
        before = cur_map(env)
        hit = any(objective.reached(env) and objective.key not in reached
                  for objective in CHAIN)
        writer.add_step(frame[..., 0], to_multihot(action), party_count(env),
                        hit, (before, *coords(env)))
        frame = env.step(action)
        handlers.note(action, cur_map(env) != before)
        previous = action
        maps.add(cur_map(env))
        for objective in CHAIN:
            if objective.reached(env):
                reached.add(objective.key)

    writer.add_step(frame[..., 0], to_multihot(ACTIONS.index("noop")),
                    party_count(env), False, (cur_map(env), *coords(env)))
    species = party_species(env)
    writer.end_episode(kind, name, True, -1 if species is None else species, -1)
    return {"maps": sorted(m for m in maps if m is not None)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rom", default="pokeyellow.gbc")
    parser.add_argument("--out", default="episodes/part2.h5")
    parser.add_argument("--per-state", type=int, default=40)
    parser.add_argument("--actions", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    jobs = episode_plan(MILESTONE_STATES, args.per_state, args.seed)
    covered = set()
    with EpisodeWriter(args.out, args.rom, MILESTONE_STATES[0], act_frames=60,
                       seed=args.seed, game="yellow", task="yellow_part2") as writer:
        for index, job in enumerate(jobs, start=1):
            env = PokeEnv(args.rom, job.state)
            # No settle(): it shifts the frame phase, and these episodes are
            # the record of what a given button sequence produces.
            report = record_episode(env, writer, args.actions, job.seed, job.kind,
                                    f"{Path(job.state).stem}-{job.seed}")
            env.close()
            covered.update(report["maps"])
            print(f"{index}/{len(jobs)} {job.kind:7s} {job.state} maps={report['maps']}",
                  flush=True)
    print("maps covered:", sorted(covered))


if __name__ == "__main__":
    main()
