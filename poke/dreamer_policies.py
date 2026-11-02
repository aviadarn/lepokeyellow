"""Behaviour policies for collecting PokeDreamer-schema data.

PokeDreamer mixes a frozen PPO agent with random exploration. The repo's
trained policies do not transfer here -- they read DMG grey frames at a
60-frame step, and this data is colour at 24 ticks -- so the mix is:

  * STICKY_RANDOM: a random button, repeated for a geometric number of steps.
    Cheap coverage of menus and the button prior.
  * FRONTIER: walks towards the least-visited neighbouring tile, learning
    walls from moves that went nowhere, and mostly presses A through text
    and battles. Gets out of rooms and across maps, which pure random does
    not (a random walk spends 1,500 steps not finding Pikachu).

Neither policy ever rolls the emulator back. A lookahead would make the data
look better and break the one guarantee the dataset makes: replaying
`actions` from the episode's start state reproduces `obs`. The oracle's own
notes record that save/load is not bit-exact for everything.
"""

from __future__ import annotations

import collections

import numpy as np

STICKY_RANDOM, FRONTIER = 0, 1
POLICY_NAMES = {STICKY_RANDOM: "sticky_random", FRONTIER: "frontier"}

DOWN, LEFT, RIGHT, UP, A, B, START, PASS = range(8)
DIRECTIONS = {DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1), UP: (-1, 0)}

#: Random-button prior: directions dominate, START is rare because the menu
#: eats a random walker, and PASS keeps some idle frames in the data.
RANDOM_WEIGHTS = np.array([0.16, 0.16, 0.16, 0.16, 0.20, 0.10, 0.03, 0.03])


class StickyRandom:
    policy_id = STICKY_RANDOM

    def __init__(self, rng: np.random.Generator, repeat: float = 0.5):
        self.rng = rng
        self.repeat = repeat
        self.last = int(rng.choice(8, p=RANDOM_WEIGHTS))

    def __call__(self, fields: dict) -> int:
        if self.rng.random() >= self.repeat:
            self.last = int(self.rng.choice(8, p=RANDOM_WEIGHTS))
        return self.last


class Frontier:
    """Novelty-seeking walker over (map, y, x), with no emulator lookahead."""

    policy_id = FRONTIER

    def __init__(self, rng: np.random.Generator, epsilon: float = 0.15,
                 momentum: float = 0.6):
        self.rng = rng
        self.epsilon = epsilon
        self.momentum = momentum
        self.visits: collections.Counter = collections.Counter()
        self.blocked: set[tuple] = set()
        self.previous: tuple | None = None
        self.last = DOWN

    def __call__(self, fields: dict) -> int:
        tile = (int(fields["map_ids"]), int(fields["ys"]), int(fields["xs"]))
        free = not fields["in_battles"] and not fields["dialog_opens"]
        # A direction that left the player on the same tile, with nothing on
        # screen to explain it, is a wall from that tile.
        if self.previous == tile and self.last in DIRECTIONS and free:
            self.blocked.add((tile, self.last))
        self.previous = tile
        self.visits[tile] += 1

        if not free:
            # Text and battle menus: mostly A, with enough of B and the d-pad
            # to leave a menu or pick another move now and then. Up and left
            # outweigh down and right because FIGHT is the battle menu's top
            # left: without LEFT the first version could reach RUN but never
            # get back, and spent 99% of an episode in one trainer battle
            # choosing RUN ("No! There's no running from a trainer battle!").
            self.last = int(self.rng.choice([A, B, UP, LEFT, DOWN, RIGHT],
                                            p=[0.6, 0.1, 0.09, 0.09, 0.06, 0.06]))
            return self.last
        if self.rng.random() < self.epsilon:
            self.last = int(self.rng.choice(8, p=RANDOM_WEIGHTS))
            return self.last
        options = [d for d in DIRECTIONS if (tile, d) not in self.blocked]
        if not options:
            self.blocked = {key for key in self.blocked if key[0] != tile}
            options = list(DIRECTIONS)
        if self.last in options and self.rng.random() < self.momentum:
            neighbour = self._step(tile, self.last)
            if self.visits[neighbour] <= min(self.visits[self._step(tile, d)]
                                             for d in options):
                return self.last
        counts = np.array([self.visits[self._step(tile, d)] for d in options])
        best = [d for d, c in zip(options, counts) if c == counts.min()]
        self.last = int(self.rng.choice(best))
        return self.last

    @staticmethod
    def _step(tile, direction):
        dy, dx = DIRECTIONS[direction]
        return (tile[0], tile[1] + dy, tile[2] + dx)


def make(policy_id: int, rng: np.random.Generator):
    return {STICKY_RANDOM: StickyRandom, FRONTIER: Frontier}[policy_id](rng)
