"""Waypoints in tiles, taken from the demonstration's recorded positions.

The latent ladder had to guess whether a rung was cleared, because nothing in
the representation says "two thirds of the way to the door". Waypoints do not
have that problem: the demonstration recorded (map, y, x) for every frame, so a
waypoint is a tile, and whether the player has reached it is read straight out
of RAM.

That splits the two jobs cleanly. Deciding we have arrived is ground truth.
Choosing the buttons is the model's imagination, scored by where it thinks the
player will end up. A failure can then only be the planner's.
"""

from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np

from poke.ladder import demo_episode


@dataclass(frozen=True)
class Waypoint:
    map_id: int
    y: int
    x: int
    route_step: int

    def reached(self, here, tolerance: int = 0) -> bool:
        """Same map, and within `tolerance` tiles.

        Exact matching does not work: the probe itself is only accurate to
        about 1.5 tiles, and one action is a 20-frame button hold that
        sometimes walks two tiles, so a plan can step straight over the target.
        Demanding the exact tile scored 1 waypoint out of 14 while the player
        was demonstrably walking the route.
        """
        map_id, y, x = here
        if map_id != self.map_id or y is None or x is None:
            return False
        return abs(y - self.y) + abs(x - self.x) <= tolerance

    def __str__(self) -> str:
        return f"map {self.map_id} ({self.y},{self.x})"


def localise(points, here, floor: int = 0) -> int | None:
    """Index of the route node closest to the player's ACTUAL tile.

    This is the fix for all three controller failures at once. Advancing on a
    timer strands the target; re-targeting forwards cannot recover from walking
    into the wrong room; re-targeting anywhere oscillates. Each one chooses the
    target independently of where the player is. Projecting the real position
    onto the route first makes the target a consequence of ground truth, which
    is the pure-pursuit idea from path following.

    `floor` refuses to localise behind the furthest node already reached, so a
    coincidental match on a tile the route visits twice cannot walk the plan
    backwards -- 95% of this route's frames sit on a revisited tile.
    """
    map_id, y, x = here
    if y is None or x is None:
        return None
    candidates = [
        (abs(point.y - y) + abs(point.x - x), index)
        for index, point in enumerate(points)
        if point.map_id == map_id and index >= floor
    ]
    if not candidates:
        # Off the route's remaining maps: allow a look behind the floor rather
        # than return nothing, since being in an earlier room IS the situation
        # that needs recovering from.
        candidates = [
            (abs(point.y - y) + abs(point.x - x), index)
            for index, point in enumerate(points)
            if point.map_id == map_id
        ]
    if not candidates:
        return None
    # Ties go to the later node, biasing forward along the route.
    return max(index for distance, index in candidates
               if distance == min(candidates)[0])


def build(data_path: str, stride: int = 8) -> list[Waypoint]:
    """Every `stride`-th distinct tile along the demonstration, in order.

    Consecutive duplicates are dropped first: the route stands still for long
    stretches of dialogue, and a waypoint already satisfied on arrival teaches
    the planner nothing.
    """
    start, length, _ = demo_episode(data_path)
    with h5py.File(data_path, "r") as file:
        if "location" not in file:
            raise RuntimeError(
                f"{data_path} has no location dataset; re-collect with the "
                "current scripts/collect.py (schema 3)."
            )
        location = file["location"][start : start + length]

    moves = [
        (tuple(row), step)
        for step, row in enumerate(location.tolist())
        if step == 0 or row != location[step - 1].tolist()
    ]
    picked = moves[stride - 1 :: stride]
    if moves and (not picked or picked[-1] is not moves[-1]):
        picked = picked + [moves[-1]]

    return [
        Waypoint(map_id=int(tile[0]), y=int(tile[1]), x=int(tile[2]), route_step=step)
        for tile, step in picked
        if tile[0] >= 0
    ]
