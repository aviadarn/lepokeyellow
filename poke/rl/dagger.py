"""Collect the states a learner actually reaches, so an expert can label them.

TRAINING ONLY: snapshotting is banned at play time, and everything here
snapshots. The policy being probed still plays under the contract -- the probe
wraps the *policy*, holds the real env in a closure of its own, and hands the
policy the same guarded env it would have been given otherwise.

Why this exists. Behaviour cloning on one optimal trajectory per objective fits
to zero loss and still fails: `results/part2_clone_eval.json` has the clone
walking the route correctly for five decisions and then oscillating between two
tiles for the remaining four hundred actions. The dataset contains no frame
showing the way back from a mistake, because the oracle never made one, so six
decisions in the clone is looking at a screen its training set does not hold.
DAgger closes that: roll the learner out, ask the expert what it should have
done from the states the learner itself reached, add those, retrain.

This module is the collection half. `scripts/dagger.py` is the loop, and the
expert is `poke.oracle`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from poke.objectives import current
from poke.oracle import story_key


@dataclass
class Visit:
    """One state the learner reached, and enough to relabel it later."""

    #: An emulator save state. Training-time only, and about 174 KiB.
    blob: object
    #: `poke.oracle.story_key` -- position, facing, and a digest of the plot.
    key: tuple
    #: Which policy turn of the rollout this was.
    turn: int
    #: Read from event flags, which is what the contract allows.
    objective: str
    #: How many later turns came back to this same state. The absorbing
    #: states of a stuck rollout are the ones with a count in the hundreds.
    revisits: int = 0
    #: (frame, action) for the learner's own turns just before this one.
    #: A relabelled episode replayed from a fresh GRU and a fresh frame
    #: window is a different input than the same game state is at play time;
    #: these warm both. See `poke.rl.clone.unpack`.
    context: list = field(default_factory=list)


class SnapshotProbe:
    """A policy, wrapped so the state behind each new decision is kept.

    Recording by wrapping rather than by editing `scripts/play2.play` is the
    same choice `poke.record` made for the planner: the loop that measures the
    baselines and the eval must be the loop the rollout runs, or the states
    collected are not the states the learner reaches.

    A state is snapshotted the first time its `story_key` appears and never
    again. The clone's measured failure is an oscillation between two tiles,
    and a blob per turn would have been 1,432 copies of two states -- a quarter
    of a gigabyte. `revisits` counts what the dedupe dropped, which is itself
    the measurement of how stuck the rollout was.
    """

    def __init__(self, policy, env, key=None, context: int = 16):
        self.policy = policy
        # Held here, never passed on: the policy gets whatever play() hands it.
        self._env = env
        self._key = key or (lambda env: story_key(env, with_facing=True))
        self.visits: list[Visit] = []
        self._seen: dict = {}
        # 16 and not the whole prefix: a frame is 23 KiB and a blob 174 KiB,
        # so the cap is what keeps a long rollout's visits inside a gigabyte,
        # and a GRU warmed on sixteen oscillating steps is an approximation
        # of one warmed on four hundred, not the same thing.
        self._context: deque = deque(maxlen=max(context, 0))
        self.turns = 0
        self.revisits = 0

    def __call__(self, guarded, objective_one_hot):
        key = self._key(self._env)
        if key in self._seen:
            self.revisits += 1
            self._seen[key].revisits += 1
        else:
            visit = Visit(blob=self._env.snapshot(), key=key, turn=self.turns,
                          objective=current(self._env).key,
                          context=list(self._context))
            self._seen[key] = visit
            self.visits.append(visit)
        frame = np.ascontiguousarray(self._env.screen()[:, :, 0])
        action = self.policy(guarded, objective_one_hot)
        self._context.append((frame, action))
        self.turns += 1
        return action


def spread(items, count: int) -> list:
    """`count` items evenly spaced over `items`, both ends included.

    One oracle search costs about five minutes, so a round relabels a handful
    of states rather than all of them. Even spacing rather than "the last N"
    on purpose: the end is where the learner is stuck, and the start is the
    part it still gets right -- a round that only ever labels failures teaches
    the clone nothing about the prefix it would otherwise keep.
    """
    if count <= 0:
        return []
    if count >= len(items):
        return list(items)
    if count == 1:
        return [items[0]]
    step = (len(items) - 1) / (count - 1)
    return [items[round(index * step)] for index in range(count)]


def by_time_spent(visits, count: int) -> list:
    """The `count` states the learner spent the most turns in, in turn order.

    `spread` was the first rule and the measurement killed it: five of round
    2's six relabelled states came from turns 0 to 14, because almost every
    distinct state of a stuck rollout happens before it gets stuck. Spreading
    over distinct states therefore spends the search budget on the prefix the
    clone already walks correctly.

    Time spent is the right weight. A state the rollout returned to three
    hundred times is the absorbing one, and every solution found from it is a
    full path to the goal, so the route beyond it is re-taught anyway. Ties go
    to the later state, which is the further one along the route.
    """
    if count <= 0:
        return []
    ranked = sorted(visits, key=lambda visit: (visit.revisits, visit.turn),
                    reverse=True)[:count]
    return sorted(ranked, key=lambda visit: visit.turn)
