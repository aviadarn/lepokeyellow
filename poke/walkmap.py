"""Walkable-tile graph, learned from the positions already recorded.

The planner's cost was Manhattan distance to the target tile, which assumes an
empty room. Pallet Town is not empty: the state the planner starts from is
standing outside the player's own front door, and the tile due north is the
doorway back INTO the house. The Oak trigger sits north-east, so a greedy
"reduce |dy| + |dx|" heuristic walks into a wall and stays there -- measured at
0/3 seeds across every horizon and action-repeat setting, with the player never
leaving the y it started on.

No new data is needed to fix it. The dataset records (map, y, x) for every
frame, so consecutive frames within an episode are observed transitions: this
tile connects to that one. BFS from the target over those edges gives a
distance that respects walls, doors and one-way steps, and the planner can be
scored on that instead.

What the graph cannot say is whether an unvisited tile is walkable. Unknown
tiles are not treated as free space -- they get a penalty -- because guessing
the opposite is how the Manhattan heuristic failed in the first place.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import h5py
import numpy as np


@dataclass
class WalkMap:
    """Observed adjacency per map, and BFS distances on demand."""

    edges: dict[int, dict[tuple[int, int], set[tuple[int, int]]]]

    def tiles(self, map_id: int) -> set[tuple[int, int]]:
        return set(self.edges.get(map_id, {}))

    def distances(self, map_id: int, target: tuple[int, int]) -> dict[tuple[int, int], int]:
        """BFS hop counts to `target`, over observed transitions reversed.

        Edges are directed as observed, so the search runs backwards from the
        target: a tile is scored by how many observed steps reach the target
        FROM it, which is the quantity a planner needs.
        """
        graph = self.edges.get(map_id, {})
        incoming: dict[tuple[int, int], set[tuple[int, int]]] = {}
        for source, targets in graph.items():
            for destination in targets:
                incoming.setdefault(destination, set()).add(source)

        out = {target: 0}
        queue = deque([target])
        while queue:
            tile = queue.popleft()
            for previous in incoming.get(tile, ()):
                if previous not in out:
                    out[previous] = out[tile] + 1
                    queue.append(previous)
        return out

    def field(self, map_id: int, target: tuple[int, int],
              shape: tuple[int, int], unknown: float) -> np.ndarray:
        """Distance lookup as a dense [H, W] array, for scoring on the GPU."""
        grid = np.full(shape, unknown, dtype=np.float32)
        for (y, x), distance in self.distances(map_id, target).items():
            if 0 <= y < shape[0] and 0 <= x < shape[1]:
                grid[y, x] = float(distance)
        return grid


def build(data_path: str) -> WalkMap:
    with h5py.File(data_path, "r") as file:
        if "location" not in file:
            raise RuntimeError(
                f"{data_path} has no location dataset; re-collect with the "
                "current scripts/collect.py (schema 3)."
            )
        location = file["location"][:]
        episodes = file["episodes"][:]

    edges: dict[int, dict[tuple[int, int], set[tuple[int, int]]]] = {}
    for start, length in episodes:
        span = location[int(start) : int(start) + int(length)]
        for before, after in zip(span[:-1], span[1:]):
            # Same map, both readable, and the player actually moved.
            if before[0] != after[0] or before[0] < 0 or before[1] < 0 or after[1] < 0:
                continue
            source = (int(before[1]), int(before[2]))
            destination = (int(after[1]), int(after[2]))
            if source == destination:
                continue
            edges.setdefault(int(before[0]), {}).setdefault(source, set()).add(destination)
            edges[int(before[0])].setdefault(destination, set())
    return WalkMap(edges)
