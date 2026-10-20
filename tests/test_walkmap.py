import numpy as np
import pytest

from poke.walkmap import WalkMap, build
from poke.writer import EpisodeWriter


@pytest.fixture
def h5_path(tmp_path):
    rom = tmp_path / "f.gb"; rom.write_bytes(b"\x00" * 1024)
    state = tmp_path / "s.state"; state.write_bytes(b"\x01" * 16)
    route = tmp_path / "r.json"; route.write_text("{}")
    return tmp_path / "poke.h5", rom, state, route


def write(h5_path, episodes):
    out, rom, state, route = h5_path
    with EpisodeWriter(out, rom, state, 60, 0, game="yellow", task="yellow_pikachu",
                       route=route) as w:
        for index, locations in enumerate(episodes):
            for step, loc in enumerate(locations, start=1):
                w.add_step(np.full((144, 160), step % 251, np.uint8),
                           np.zeros(6, np.uint8), 0, False, loc)
            w.end_episode("random", f"r{index}", False, -1, -1)
    return str(out)


def test_edges_come_from_observed_transitions(h5_path):
    path = write(h5_path, [[(0, 5, 5), (0, 6, 5), (0, 6, 6)]])
    walk = build(path)
    assert walk.edges[0][(5, 5)] == {(6, 5)}
    assert walk.edges[0][(6, 5)] == {(6, 6)}


def test_standing_still_is_not_an_edge(h5_path):
    path = write(h5_path, [[(0, 5, 5), (0, 5, 5), (0, 6, 5)]])
    assert build(path).edges[0][(5, 5)] == {(6, 5)}


def test_map_changes_do_not_create_edges(h5_path):
    """A doorway is a transition between maps, not an edge inside one, so the
    map it left has no intra-map edge to record at all."""
    path = write(h5_path, [[(0, 6, 5), (37, 7, 2), (37, 5, 2)]])
    walk = build(path)
    assert walk.tiles(0) == set()
    assert walk.edges[37][(7, 2)] == {(5, 2)}


def test_unreadable_positions_are_skipped(h5_path):
    """A -1 read must not become a tile, and must not bridge the two real ones
    either -- the player did not teleport, the read failed."""
    path = write(h5_path, [[(0, 5, 5), (0, -1, -1), (0, 6, 5)]])
    walk = build(path)
    assert (-1, -1) not in walk.tiles(0)
    assert walk.edges.get(0, {}).get((5, 5), set()) == set()


def test_distances_follow_observed_edges_not_geometry():
    """The whole point: a tile one step away in Manhattan terms can be many
    steps away through the actual walkable space, or unreachable."""
    walk = WalkMap({0: {
        (5, 5): {(5, 6)}, (5, 6): {(5, 7)}, (5, 7): {(4, 7)},
        (4, 7): {(3, 7)}, (3, 7): set(), (4, 5): set(),
    }})
    distances = walk.distances(0, (3, 7))
    assert distances[(3, 7)] == 0
    assert distances[(5, 5)] == 4
    # Manhattan-adjacent to the target but never observed connecting to it.
    assert (4, 5) not in distances


def test_distances_respect_one_way_edges():
    walk = WalkMap({0: {(1, 1): {(2, 2)}, (2, 2): set()}})
    assert walk.distances(0, (2, 2))[(1, 1)] == 1
    assert (2, 2) not in walk.distances(0, (1, 1)) or \
        walk.distances(0, (1, 1)).get((2, 2)) is None


def test_field_marks_unvisited_tiles_as_unknown_not_free():
    """Treating never-seen tiles as free space is exactly how the Manhattan
    heuristic walked into the house."""
    walk = WalkMap({0: {(0, 0): {(0, 1)}, (0, 1): set()}})
    grid = walk.field(0, (0, 1), (3, 3), unknown=99.0)
    assert grid[0, 1] == 0.0 and grid[0, 0] == 1.0
    assert grid[2, 2] == 99.0


def test_missing_location_dataset_is_a_clear_error(h5_path):
    import h5py as h5

    path = write(h5_path, [[(0, 5, 5), (0, 6, 5)]])
    with h5.File(path, "a") as f:
        del f["location"]
    with pytest.raises(RuntimeError, match="no location dataset"):
        build(path)
