import h5py
import numpy as np
import pytest

from poke.waypoints import Waypoint, build
from poke.writer import EpisodeWriter


@pytest.fixture
def h5_path(tmp_path):
    rom = tmp_path / "f.gb"; rom.write_bytes(b"\x00" * 1024)
    state = tmp_path / "s.state"; state.write_bytes(b"\x01" * 16)
    route = tmp_path / "r.json"; route.write_text("{}")
    return tmp_path / "poke.h5", rom, state, route


def write(h5_path, locations, kind="demo"):
    out, rom, state, route = h5_path
    with EpisodeWriter(out, rom, state, 60, 0, game="yellow", task="yellow_pikachu",
                       route=route) as w:
        for step, loc in enumerate(locations, start=1):
            w.add_step(np.full((144, 160), step % 251, np.uint8),
                       np.zeros(6, np.uint8), 0, step == len(locations), loc)
        w.end_episode(kind, "d0", True, -1, len(locations))
    return str(out)


def test_waypoints_are_tiles_in_route_order(h5_path):
    path = write(h5_path, [(38, 6, 3), (38, 7, 3), (38, 8, 3), (37, 1, 2)])
    points = build(path, stride=1)
    assert [(w.map_id, w.y, w.x) for w in points] == [
        (38, 6, 3), (38, 7, 3), (38, 8, 3), (37, 1, 2)
    ]


def test_standing_still_does_not_create_waypoints(h5_path):
    """The route stops for long stretches of dialogue. A waypoint already
    satisfied on arrival teaches the planner nothing."""
    path = write(h5_path, [(38, 6, 3)] * 20 + [(38, 7, 3)] * 20)
    assert len(build(path, stride=1)) == 2


def test_stride_subsamples_but_keeps_the_last(h5_path):
    path = write(h5_path, [(38, 1, i) for i in range(10)])
    points = build(path, stride=4)
    assert len(points) < 10
    assert (points[-1].y, points[-1].x) == (1, 9)


def test_unreadable_tiles_are_dropped(h5_path):
    path = write(h5_path, [(38, 6, 3), (-1, -1, -1), (38, 7, 3)])
    assert all(w.map_id >= 0 for w in build(path, stride=1))


def test_reached_compares_the_whole_tile(h5_path):
    point = Waypoint(38, 6, 3, 0)
    assert point.reached((38, 6, 3))
    assert not point.reached((37, 6, 3))
    assert not point.reached((38, 6, 4))


def test_missing_location_dataset_is_a_clear_error(tmp_path, h5_path):
    import poke.writer as writer_module

    out, rom, state, route = h5_path
    path = write(h5_path, [(38, 6, 3), (38, 7, 3)])
    with h5py.File(path, "a") as f:
        del f["location"]
    with pytest.raises(RuntimeError, match="no location dataset"):
        build(path, stride=1)


def test_tolerance_accepts_a_near_miss_on_the_same_map():
    """The probe is good to ~1.5 tiles and one action can walk two, so exact
    matching scored 1 waypoint of 14 while the player was walking the route."""
    point = Waypoint(38, 6, 3, 0)
    assert not point.reached((38, 6, 5))
    assert point.reached((38, 6, 4), tolerance=1)
    assert point.reached((38, 7, 4), tolerance=2)


def test_tolerance_never_crosses_maps():
    point = Waypoint(38, 6, 3, 0)
    assert not point.reached((37, 6, 3), tolerance=5)


def test_unreadable_coordinates_are_not_an_arrival():
    assert not Waypoint(38, 6, 3, 0).reached((38, None, None), tolerance=5)


def points():
    return [
        Waypoint(38, 6, 3, 0), Waypoint(38, 1, 6, 10),
        Waypoint(37, 1, 5, 20), Waypoint(37, 7, 2, 30),
        Waypoint(0, 5, 5, 40), Waypoint(0, 0, 10, 50),
        Waypoint(40, 11, 5, 60),
    ]


def test_localise_projects_onto_the_nearest_node_on_the_same_map():
    from poke.waypoints import localise

    assert localise(points(), (37, 1, 5)) == 2
    assert localise(points(), (0, 0, 9)) == 5


def test_localise_ignores_other_maps_even_when_coordinates_match():
    """(6,3) in the bedroom and (6,3) in Pallet Town are unrelated places."""
    from poke.waypoints import localise

    assert localise(points(), (0, 6, 3)) == 4


def test_localise_respects_the_floor_so_it_cannot_walk_backwards():
    """95% of this route's frames sit on a tile it revisits, so a coincidental
    match must not drag the plan back."""
    from poke.waypoints import localise

    assert localise(points(), (38, 6, 3)) == 0
    assert localise(points(), (38, 6, 3), floor=1) == 1


def test_localise_looks_behind_the_floor_when_the_player_has_regressed():
    """Walking back into the house leaves no node at or after the floor on that
    map; returning None would strand the plan, which is the bug being fixed."""
    from poke.waypoints import localise

    assert localise(points(), (38, 1, 6), floor=4) == 1


def test_localise_returns_none_off_the_route_entirely():
    from poke.waypoints import localise

    assert localise(points(), (12, 3, 3)) is None
    assert localise(points(), (38, None, None)) is None


def test_localise_breaks_ties_forwards():
    from poke.waypoints import localise

    same = [Waypoint(38, 2, 2, 0), Waypoint(38, 2, 2, 5)]
    assert localise(same, (38, 2, 2)) == 1


def test_the_exit_node_of_a_room_is_its_last_one_on_the_route():
    """When the target is in another room the planner has no gradient, so it
    should steer at the route's exit from the current room instead."""
    route = points()
    at, current_map = 2, 37
    exits = [i for i in range(at, len(route)) if route[i].map_id == current_map]
    assert max(exits) == 3
    assert (route[3].y, route[3].x) == (7, 2)


def test_being_nearest_is_not_the_same_as_having_arrived():
    """localise returns the closest node at ANY distance. Counting that as
    progress reported node 22 of 38 while the player stood 8 tiles away and
    never entered the lab."""
    from poke.waypoints import localise

    route = points()
    here = (0, 9, 18)
    at = localise(route, here)
    assert at is not None
    assert not route[at].reached(here, tolerance=1)


def test_facing_is_the_direction_the_player_arrived_from(h5_path):
    """A trigger needs the tile AND the way the player is facing: Pikachu's
    hand-over is a ball on a table, and reaching its tile facing east does
    nothing at all."""
    path = write(h5_path, [(40, 5, 7), (40, 4, 7), (40, 4, 8)])
    points = build(path, stride=1)
    assert [p.facing for p in points] == [None, "up", "right"]


def test_facing_is_unknown_when_the_move_cannot_be_a_turn(h5_path):
    """A cutscene moves the player diagonally, several tiles at once, or into
    another room. None of those say which way they ended up facing."""
    path = write(h5_path, [(40, 5, 7), (40, 4, 6), (0, 4, 6)])
    assert [p.facing for p in build(path, stride=1)] == [None, None, None]


def gated_route():
    return [
        Waypoint(40, 5, 5, 0),
        Waypoint(40, 4, 7, 10, scripted=True, facing="up"),
        Waypoint(40, 4, 5, 20, scripted=True),
        Waypoint(40, 3, 5, 30, scripted=True, facing="up"),
    ]


def test_a_trigger_tile_cannot_be_skipped():
    """The route holds (3, 5) twice -- carried in through the door, and carried
    there again by the hand-over -- and localise matches the later one because
    ties go forwards. Measured: the planner walked onto it under its own power
    at action 111, reported 38/38 complete, and never stood on the tile that
    starts the scene."""
    from poke.waypoints import gate

    assert gate(gated_route(), 3, touched=set()) == 1
    assert gate(gated_route(), 3, touched={1}) == 2
    assert gate(gated_route(), 3, touched={1, 2}) == 3


def test_the_gate_is_open_once_every_trigger_has_been_stood_on():
    from poke.waypoints import gate

    assert gate(gated_route(), 3, touched={1, 2, 3}) == 3


def test_the_gate_never_drags_the_route_forwards():
    """It is a ceiling, not a target: a player still at the first node stays
    there even though the gate is further on."""
    from poke.waypoints import gate

    assert gate(gated_route(), 0, touched=set()) == 0
