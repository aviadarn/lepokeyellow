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
tiles are not treated as free space -- they get a penalty that grows with how
far outside the observed region they sit -- because guessing the opposite is
how the Manhattan heuristic failed in the first place.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import h5py
import numpy as np


def field_factory():
    return field(default_factory=dict)


def _grid(distances: dict[tuple[int, int], float], shape: tuple[int, int],
          unknown: float, fill: float) -> np.ndarray:
    """Dense [H, W] costs from a tile -> distance mapping."""
    grid = np.full(shape, unknown, dtype=np.float32)
    known = [((y, x), float(d)) for (y, x), d in distances.items()
             if 0 <= y < shape[0] and 0 <= x < shape[1]]
    if not known:
        return grid
    for (y, x), distance in known:
        grid[y, x] = distance
    if fill <= 0:
        return grid

    coordinates = np.array([tile for tile, _ in known], dtype=np.float32)
    values = np.array([d for _, d in known], dtype=np.float32)
    rows, cols = np.nonzero(grid >= unknown)
    if len(rows):
        offsets = (np.abs(rows[:, None] - coordinates[None, :, 0])
                   + np.abs(cols[:, None] - coordinates[None, :, 1]))
        grid[rows, cols] = np.minimum(
            (values[None, :] + fill * offsets).min(axis=1), unknown
        )
    return grid


@dataclass
class WalkMap:
    """Observed adjacency per map, plus the doors between maps.

    `edges` is adjacency inside one map, keyed by map id. `doors` holds the
    transitions that change map, keyed by (map, y, x) in both directions as
    observed. Keeping them apart is not bookkeeping for its own sake: a cost
    field over one map alone has no value for any tile in another, so a planner
    scored on it is blind the moment it steps through a doorway -- measured, the
    planner walked into the player's house and then sat there for 38 straight
    actions because every action inside looked identical and terrible
    (results/nav_decisions_mlp_both.json). `global_distances` searches across
    doors so there is a way back out.
    """

    edges: dict[int, dict[tuple[int, int], set[tuple[int, int]]]]
    doors: dict[tuple[int, int, int], set[tuple[int, int, int]]] = field_factory()

    def tiles(self, map_id: int) -> set[tuple[int, int]]:
        return set(self.edges.get(map_id, {}))

    def walkable(self, map_id: int) -> set[tuple[int, int]]:
        """Every tile the player was ever observed standing on, in this map.

        This is the free-space map, and it is a better basis for a distance
        than the recorded transitions are. One recorded step holds a button for
        20 frames and then settles for 40, which is long enough to cross more
        than one tile, so the transitions include diagonal and two-tile jumps
        and a BFS over them does not count in units of anything -- measured,
        the resulting field was not monotone toward the target, with (5, 5) at
        6 hops, (6, 6) at 5 and (5, 6) at 7, so descending it greedily gets
        stuck on the field rather than on the model. Four-adjacency restricted
        to observed tiles gives one hop per tile.
        """
        tiles = set(self.edges.get(map_id, {}))
        for (y, x), destinations in self.edges.get(map_id, {}).items():
            tiles.update(destinations)
        for (m, y, x), destinations in self.doors.items():
            if m == map_id:
                tiles.add((y, x))
            for (dm, dy, dx) in destinations:
                if dm == map_id:
                    tiles.add((dy, dx))
        return tiles

    def nodes(self) -> set[tuple[int, int, int]]:
        out = {(map_id, y, x) for map_id, graph in self.edges.items()
               for (y, x) in graph}
        for source, destinations in self.doors.items():
            out.add(source)
            out.update(destinations)
        return out

    def global_distances(
        self, target: tuple[int, int, int], adjacency: str = "grid",
        blocked=(), two_way_doors: bool = False,
    ) -> dict[tuple[int, int, int], int]:
        """BFS hop counts to a (map, y, x) node, across maps.

        `adjacency="grid"` steps between four-adjacent observed tiles, so one
        hop is one tile. `"observed"` uses the recorded transitions as they
        are, which is what the data literally says but counts a two-tile walk
        as one hop; see `walkable`.

        Edges run backwards from the target, as in `distances`: a node is
        scored by how many steps reach the target from it.

        `blocked` drops (source, destination) pairs the caller has since found
        to be walls. Four-adjacency is an assumption -- both tiles were stood
        on, so you can step between them -- and inside Oak's lab it is false:
        (4, 4) and (4, 5) are both recorded, a counter runs between them, and
        the only reason (4, 5) is in the data at all is that a cutscene carried
        the player onto it. The field read one hop east to the trigger tile,
        the planner pressed right 33 times into the counter, and the run
        stalled three tiles from Pikachu.
        """
        if adjacency not in ("grid", "observed"):
            raise ValueError(f"unknown adjacency {adjacency!r}")
        incoming: dict[tuple[int, int, int], set[tuple[int, int, int]]] = {}
        if adjacency == "grid":
            for map_id in set(self.edges) | {m for (m, _, _) in self.doors}:
                tiles = self.walkable(map_id)
                for (y, x) in tiles:
                    for (ny, nx) in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                        if (ny, nx) in tiles:
                            incoming.setdefault((map_id, ny, nx), set()).add(
                                (map_id, y, x)
                            )
                # Plus the recorded transitions themselves. A cutscene can move
                # the player diagonally or several tiles at once along a path
                # that cannot be reconstructed, and dropping those would leave
                # the graph less connected than the data -- which is how the
                # waypoint inside the lab became unreachable from Pallet Town.
                for (y, x), destinations in self.edges.get(map_id, {}).items():
                    for (ny, nx) in destinations:
                        incoming.setdefault((map_id, ny, nx), set()).add(
                            (map_id, y, x)
                        )
        else:
            for map_id, graph in self.edges.items():
                for (y, x), destinations in graph.items():
                    for (ny, nx) in destinations:
                        incoming.setdefault((map_id, ny, nx), set()).add((map_id, y, x))
        for source, destinations in self.doors.items():
            for destination in destinations:
                incoming.setdefault(destination, set()).add(source)
                # Off by default, and measured that way round. The case for
                # it: no episode ever left Oak's lab, so every recorded
                # doorway points inwards and from inside the lab no tile in
                # Pallet has any cost at all -- the player stepped out, back in
                # through the same door, and spent 482 actions on a flat field.
                # The case against, which won: turning it on gave the planner
                # a door as a cheap shortcut between rooms it had never walked
                # that way, and two of three seeds went from four milestones
                # to none.
                if two_way_doors:
                    incoming.setdefault(source, set()).add(destination)
        for source, destination in blocked:
            incoming.get(destination, set()).discard(source)

        out = {target: 0}
        queue = deque([target])
        while queue:
            node = queue.popleft()
            for previous in incoming.get(node, ()):
                if previous not in out:
                    out[previous] = out[node] + 1
                    queue.append(previous)
        return out

    def fields(self, maps: list[int], target: tuple[int, int, int],
               shape: tuple[int, int], unknown: float,
               fill: float = 2.0, adjacency: str = "grid",
               blocked=(), two_way_doors: bool = False) -> np.ndarray:
        """One [H, W] grid per entry of `maps`, stacked: [len(maps), H, W].

        `maps` is the probe's own map order, so a decoded map index indexes the
        first axis directly and no wrong-map penalty is needed -- being in the
        wrong room is simply expensive, by however many doors it is from the
        target.
        """
        distances = self.global_distances(target, adjacency, blocked,
                                          two_way_doors)
        grids = []
        for map_id in maps:
            here = {(y, x): d for (m, y, x), d in distances.items() if m == map_id}
            grids.append(_grid(here, shape, unknown, fill))
        return np.stack(grids)

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
              shape: tuple[int, int], unknown: float,
              fill: float = 2.0) -> np.ndarray:
        """Distance lookup as a dense [H, W] array, for scoring on the GPU.

        Unvisited cells are not left as one flat plateau at `unknown`. The
        planner reads tiles off a probe with roughly half a tile of error, so
        rounding a predicted position can land one cell outside the observed
        region -- and with a flat plateau that turns a half-tile read-out error
        into a 60-hop cost error, which is how three of seven actions scored
        identically catastrophically in results/nav_resolution_mlp.json.

        So an unvisited cell costs the nearest observed cell's distance plus
        `fill` per tile of extrapolation, capped at `unknown`. Unknown ground
        is still more expensive than observed ground, which is the point of the
        penalty, but the field keeps a gradient across the boundary instead of
        a cliff.
        """
        return _grid(self.distances(map_id, target), shape, unknown, fill)


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
    doors: dict[tuple[int, int, int], set[tuple[int, int, int]]] = {}
    for start, length in episodes:
        span = location[int(start) : int(start) + int(length)]
        for before, after in zip(span[:-1], span[1:]):
            if before[0] < 0 or before[1] < 0 or after[0] < 0 or after[1] < 0:
                continue
            if before[0] != after[0]:
                doors.setdefault(
                    (int(before[0]), int(before[1]), int(before[2])), set()
                ).add((int(after[0]), int(after[1]), int(after[2])))
                continue
            source = (int(before[1]), int(before[2]))
            destination = (int(after[1]), int(after[2]))
            if source == destination:
                continue
            graph = edges.setdefault(int(before[0]), {})
            # One recorded step holds a button for 20 frames and settles for
            # 40, which is long enough to cross two tiles, and the tile in
            # between is then never recorded standing-on. Four-adjacency over
            # recorded tiles alone therefore cannot bridge it, and the graph
            # fragments: measured, only 12 tiles could reach the waypoint
            # inside Oak's lab, so the cost field over the whole of Pallet Town
            # was one flat plateau and the planner had nothing to descend. The
            # player demonstrably walked THROUGH the middle tile, so it is
            # walkable and goes in, splitting the step into its two halves.
            steps = [(source, destination)]
            dy, dx = destination[0] - source[0], destination[1] - source[1]
            if (dy == 0) != (dx == 0):
                # A straight move of n tiles is n steps through n-1 tiles the
                # player demonstrably stood on. Strides inside Oak's lab reach
                # four and seven tiles, because a cutscene walks the player
                # while the recorder samples once per action, so without this
                # the lab is a handful of disconnected islands.
                stride = abs(dy + dx)
                step = ((dy > 0) - (dy < 0), (dx > 0) - (dx < 0))
                path = [(source[0] + step[0] * n, source[1] + step[1] * n)
                        for n in range(stride + 1)]
                steps = list(zip(path[:-1], path[1:]))
            for begin, end in steps:
                graph.setdefault(begin, set()).add(end)
                graph.setdefault(end, set())
    return WalkMap(edges, doors)


def lookup(fields: np.ndarray, map_index: int | None, y, x,
           unknown: float) -> float:
    """Cost of one (map index, y, x), tolerant of an unreadable position.

    `fields` is what `WalkMap.fields` returns, so `map_index` indexes the
    probe's own map order and a map the probe does not know has no cost.
    """
    if map_index is None or y is None or x is None:
        return float(unknown)
    if not 0 <= map_index < fields.shape[0]:
        return float(unknown)
    rows, cols = fields.shape[1:]
    return float(fields[map_index,
                        min(max(int(round(y)), 0), rows - 1),
                        min(max(int(round(x)), 0), cols - 1)])
