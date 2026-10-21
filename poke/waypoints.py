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

from pathlib import Path

import h5py
import numpy as np

from poke.ladder import demo_episode


@dataclass(frozen=True)
class Waypoint:
    map_id: int
    y: int
    x: int
    route_step: int
    # True when the demonstration lost control here -- the tile the player
    # walks to in order to START a scene. Oak's encounter is the case that
    # matters: it fires at the north edge of Pallet, walks the player back
    # south, and fires again. The tile therefore keeps changing, which is
    # exactly what the "tile stopped moving" dialogue detector needs in order
    # to notice a scene, so the planner reads it as navigation and tries to
    # walk north forever. Measured: 240 actions oscillating between (1, 11)
    # and (3, 11) two tiles from the trigger, pressing b 61 times.
    scripted: bool = False
    # The direction the demonstration was moving when it stepped onto this
    # tile, and therefore the way the player is facing when the scene fires.
    # Standing on the trigger tile is not enough: Gen 1 interacts with the tile
    # the player FACES, and Pikachu's hand-over is a Poke Ball on a table one
    # tile north of (4, 7). Reached facing east, 60 A presses do nothing;
    # turn up first and the party has a Pikachu 34 presses later.
    facing: str | None = None

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


def gate(points, at: int, touched) -> int:
    """Never aim past a trigger tile that has not been stood on.

    A scripted waypoint is a tile the demonstration lost control ON -- the
    scene fires because the player walked there. The tiles the scene then
    carries them to are waypoints as well, and those are the ones a planner
    arrives at by accident. This route holds map 40 (3, 5) twice, once carried
    in through the lab door and once carried by the hand-over itself, and
    `localise` matches the later one because ties go forward.

    Measured: the planner walked onto (3, 5) under its own power at action 111,
    the route reported 38/38 complete, and the tile that actually starts the
    hand-over -- two tiles east and one south -- was never stood on at all. The
    run then spent 489 actions pressing A at a table that was not there.
    """
    for index, point in enumerate(points):
        if point.scripted and index not in touched:
            return min(at, index)
    return at


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


def build(data_path: str, stride: int = 8,
          annotations: str | None = None) -> list[Waypoint]:
    """Every `stride`-th distinct tile along the demonstration, in order.

    Consecutive duplicates are dropped first: the route stands still for long
    stretches of dialogue, and a waypoint already satisfied on arrival teaches
    the planner nothing.

    `annotations` (from scripts/annotate_route.py) drops the steps where the
    game was moving the player rather than the player moving. 42% of this route
    is scripted, including a 116-step stretch for Oak's encounter, and those
    tiles are not places you can walk to -- the planner reached one of them
    exactly and nothing happened, because arriving under your own power is not
    the same event. Excluding them leaves the real trigger tile as the last
    waypoint before the scene, which is what the planner should aim at.
    """
    start, length, _ = demo_episode(data_path)
    with h5py.File(data_path, "r") as file:
        if "location" not in file:
            raise RuntimeError(
                f"{data_path} has no location dataset; re-collect with the "
                "current scripts/collect.py (schema 3)."
            )
        location = file["location"][start : start + length]

    navigable = None
    if annotations:
        import json

        data = json.loads(Path(annotations).read_text())
        navigable = list(data["navigable"])
        scripted_at_source = [not value for value in navigable]
        # Keep the FIRST step of each scripted run. Control is lost ON the
        # trigger tile, so the counterfactual marks it scripted -- but that is
        # the tile the player must walk to, not one the scene carried them to.
        # Dropping it removed the only waypoint that starts Oak's encounter and
        # left the planner aiming past it.
        for step in range(len(navigable)):
            if not navigable[step] and (step == 0 or navigable[step - 1]):
                navigable[step] = True

    tiles = location.tolist()

    def facing_at(step: int) -> str | None:
        """Which way the player was moving when they arrived on this tile.

        Read from the raw positions rather than from the kept waypoints: the
        step before a trigger tile is usually inside the same scripted run and
        has been filtered out, and it is exactly the step that says which way
        the player turned. A move that changes map, stands still or crosses
        both axes at once says nothing about facing.
        """
        if step <= 0 or step >= len(tiles):
            return None
        before, now = tiles[step - 1], tiles[step]
        if before[0] != now[0]:
            return None
        dy, dx = now[1] - before[1], now[2] - before[2]
        if bool(dy) == bool(dx):
            return None
        if dy:
            return "up" if dy < 0 else "down"
        return "left" if dx < 0 else "right"

    def is_scripted(step: int) -> bool:
        """The step was kept despite the counterfactual calling it scripted."""
        return (navigable is not None and step < len(scripted_at)
                and scripted_at[step])

    scripted_at = [] if navigable is None else list(scripted_at_source)
    moves = [
        (tuple(row), step)
        for step, row in enumerate(location.tolist())
        if (step == 0 or row != location[step - 1].tolist())
        and (navigable is None or step >= len(navigable) or navigable[step])
    ]
    picked = moves[stride - 1 :: stride]
    if moves and (not picked or picked[-1] is not moves[-1]):
        picked = picked + [moves[-1]]

    return [
        Waypoint(map_id=int(tile[0]), y=int(tile[1]), x=int(tile[2]),
                 route_step=step, scripted=is_scripted(step),
                 facing=facing_at(step))
        for tile, step in picked
        if tile[0] >= 0
    ]
