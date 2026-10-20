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
