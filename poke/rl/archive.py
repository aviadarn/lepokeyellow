"""Go-Explore archive: where training episodes start from.

Training only -- it restores save states, which play time bans.

A cell is (objective index, map, y // CELL, x // CELL). Coarse on purpose: one
cell per tile would store a quarter of a megabyte per step and teach the policy
to start from everywhere equally, and the point of the archive is to make the
places the policy rarely reaches cheap to practise.

Sampling weight is 1 / sqrt(1 + visits), the Go-Explore rule: a cell that has
been started from four hundred times is not where the policy is weak any more.

Two refusals, both measured rather than defensive. A state captured without
control -- a text box, a battle -- starts an episode the handler owns from the
first turn, so `add` checks `has_control` and declines. And the capacity is a
byte budget in disguise: at 178,101 bytes a state, the default 2,000 cells is
about 356 MB, which the driver holds while four workers hold their own
emulators.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from poke.control import has_control
from poke.objectives import current

#: Tiles per archive cell, on each axis.
CELL = 4
#: Cells kept. 2,000 x 178,101 bytes is about 356 MB.
DEFAULT_CAPACITY = 2000


@dataclass
class Cell:
    key: tuple
    blob: bytes
    visits: int = 0
    found_at: int = 0


class Archive:
    def __init__(self, capacity: int = DEFAULT_CAPACITY, rng=None):
        self.capacity = capacity
        self.rng = rng or random.Random()
        self.cells: dict[tuple, Cell] = {}
        self.added = 0
        self.rejected_without_control = 0

    def key_for(self, env) -> tuple:
        ram = env.game.ram
        y = env.read_ram(ram.y_coord) or 0
        x = env.read_ram(ram.x_coord) or 0
        return (current(env).index, env.read_ram(ram.cur_map), y // CELL, x // CELL)

    def add(self, env, blob: bytes | None = None) -> bool:
        """Store this position if it is a new cell. True when it was new."""
        if not has_control(env):
            self.rejected_without_control += 1
            return False
        key = self.key_for(env)
        if key in self.cells:
            return False
        self.added += 1
        self.cells[key] = Cell(key=key, blob=blob if blob is not None else env.snapshot(),
                               found_at=self.added)
        if len(self.cells) > self.capacity:
            # The most-visited cell has the least left to teach.
            crowded = max(self.cells.values(), key=lambda cell: (cell.visits, -cell.found_at))
            del self.cells[crowded.key]
        return True

    def note_visit(self, key: tuple) -> None:
        cell = self.cells.get(key)
        if cell is not None:
            cell.visits += 1

    def sample(self) -> bytes | None:
        """A start state, weighted towards cells started from least."""
        if not self.cells:
            return None
        cells = list(self.cells.values())
        weights = [1.0 / ((1.0 + cell.visits) ** 0.5) for cell in cells]
        return self.rng.choices(cells, weights=weights, k=1)[0].blob

    def __len__(self) -> int:
        return len(self.cells)

    def stats(self) -> dict:
        return {
            "cells": len(self.cells),
            "maps": sorted({key[1] for key in self.cells}),
            "objectives": sorted({key[0] for key in self.cells}),
            "bytes": sum(len(cell.blob) for cell in self.cells.values()),
            "rejected_without_control": self.rejected_without_control,
        }
