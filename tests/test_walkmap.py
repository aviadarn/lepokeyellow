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
    # (7, 2) -> (5, 2) is a straight two-tile stride, so it is recorded as two
    # steps through the tile the player walked over.
    assert walk.edges[37][(7, 2)] == {(6, 2)}
    assert walk.edges[37][(6, 2)] == {(5, 2)}


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


def test_field_charges_more_for_unvisited_tiles_than_visited_ones():
    """Treating never-seen tiles as free space is how the Manhattan heuristic
    walked into the house, so they cost more -- but not as one flat plateau.

    A plateau turns the probe's half-tile error into the full `unknown`
    penalty whenever rounding lands one cell outside the observed region, and
    then every option scores the same catastrophic number. Cost grows with how
    far outside the observed region a cell sits, and still caps at `unknown`.
    """
    walk = WalkMap({0: {(0, 0): {(0, 1)}, (0, 1): set()}})
    grid = walk.field(0, (0, 1), (8, 8), unknown=8.0, fill=2.0)
    assert grid[0, 1] == 0.0 and grid[0, 0] == 1.0
    # Three tiles of extrapolation from the target, at 2 per tile.
    assert grid[2, 2] == 6.0
    # Monotone in how far outside the observed region a cell sits...
    assert grid[1, 1] < grid[2, 2]
    # ...and never worse than `unknown`, so one plateau is still the ceiling.
    assert grid[4, 4] == 8.0 and grid[7, 7] == 8.0


def test_field_can_keep_the_hard_plateau():
    walk = WalkMap({0: {(0, 0): {(0, 1)}, (0, 1): set()}})
    grid = walk.field(0, (0, 1), (3, 3), unknown=99.0, fill=0.0)
    assert grid[2, 2] == 99.0


def test_doors_connect_maps_so_there_is_a_way_back_out(h5_path):
    """A cost field over one map alone has no value for any tile in another, so
    a planner scored on it is blind the moment it walks through a door."""
    path = write(h5_path, [[(37, 7, 2), (37, 6, 2), (0, 5, 5), (0, 6, 5)]])
    walk = build(path)
    assert walk.doors[(37, 6, 2)] == {(0, 5, 5)}
    distances = walk.global_distances((0, 6, 5), adjacency="observed")
    assert distances[(0, 5, 5)] == 1
    # Inside the house, and still scored: two steps and a doorway away.
    assert distances[(37, 6, 2)] == 2
    assert distances[(37, 7, 2)] == 3


def test_a_straight_stride_is_split_into_the_tiles_it_crossed(h5_path):
    """One recorded step holds a button for 20 frames and then settles for 40,
    long enough to cross several tiles, and the tiles in between are never
    recorded standing-on. Measured on the real dataset, strides inside Oak's
    lab reach four and seven tiles, which left the lab as four disconnected
    islands and made the waypoint inside it unreachable from Pallet Town --
    so the cost field over the whole town was one flat plateau.
    """
    path = write(h5_path, [[(0, 5, 2), (0, 5, 6), (0, 1, 6)]])
    walk = build(path)
    assert (5, 4) in walk.tiles(0) and (3, 6) in walk.tiles(0)
    for adjacency in ("grid", "observed"):
        distances = walk.global_distances((0, 1, 6), adjacency=adjacency)
        # Four east then four north, counted as the eight tiles it is.
        assert distances[(0, 5, 2)] == 8


def test_an_unreconstructable_move_is_still_an_edge(h5_path):
    """A cutscene can move the player diagonally or along a path that cannot
    be inferred. Dropping those would leave the graph less connected than the
    data, so the transition is kept as a single edge."""
    path = write(h5_path, [[(0, 5, 5), (0, 3, 2)]])
    walk = build(path)
    assert walk.edges[0][(5, 5)] == {(3, 2)}
    assert walk.global_distances((0, 3, 2), adjacency="grid")[(0, 5, 5)] == 1


def test_fields_stack_in_the_probes_map_order(h5_path):
    path = write(h5_path, [[(37, 7, 2), (37, 6, 2), (0, 5, 5), (0, 5, 6)]])
    walk = build(path)
    grids = walk.fields([0, 37], (0, 5, 6), (8, 8), unknown=99.0, fill=0.0)
    assert grids.shape == (2, 8, 8)
    assert grids[0][5, 6] == 0.0 and grids[0][5, 5] == 1.0
    assert grids[1][6, 2] == 2.0
    # Reversing the order reverses the first axis, because that axis is the
    # decoded map index and nothing else.
    assert np.array_equal(walk.fields([37, 0], (0, 5, 6), (8, 8), 99.0, 0.0),
                          grids[::-1])


def test_lookup_is_tolerant_of_an_unreadable_position():
    from poke.walkmap import lookup

    fields = np.zeros((2, 4, 4), dtype=np.float32)
    assert lookup(fields, None, 1, 1, 7.0) == 7.0
    assert lookup(fields, 0, None, 1, 7.0) == 7.0
    assert lookup(fields, 5, 1, 1, 7.0) == 7.0
    assert lookup(fields, 1, 99, 99, 7.0) == 0.0


def test_missing_location_dataset_is_a_clear_error(h5_path):
    import h5py as h5

    path = write(h5_path, [[(0, 5, 5), (0, 6, 5)]])
    with h5.File(path, "a") as f:
        del f["location"]
    with pytest.raises(RuntimeError, match="no location dataset"):
        build(path)


def test_a_wall_found_by_walking_into_it_drops_the_edge():
    """Four-adjacency assumes two recorded tiles side by side are connected.
    Inside Oak's lab that is false: (4, 4) and (4, 5) are both in the data, a
    counter runs between them, and the only reason (4, 5) was ever recorded is
    that a cutscene carried the player onto it. The field read one hop to the
    trigger tile and the planner pressed right 33 times into the counter.
    """
    walk = WalkMap({40: {
        (4, 4): set(), (4, 5): set(), (4, 6): set(), (4, 7): set(),
        (5, 4): set(), (5, 5): set(), (5, 6): set(), (5, 7): set(),
    }})
    target = (40, 4, 7)
    assert walk.global_distances(target)[(40, 4, 4)] == 3
    # The counter: you cannot step east off (4, 4).
    blocked = {((40, 4, 4), (40, 4, 5))}
    around = walk.global_distances(target, blocked=blocked)
    assert around[(40, 4, 4)] == 5          # down, along row 5, back up
    assert around[(40, 4, 6)] == 1          # everything else is untouched


def test_blocking_an_edge_leaves_the_other_direction_alone():
    """A wall is learned from one side. Nothing says the reverse step is
    impossible, and the graph already carries one-way edges."""
    walk = WalkMap({0: {(1, 1): set(), (1, 2): set()}})
    blocked = {((0, 1, 1), (0, 1, 2))}
    assert walk.global_distances((0, 1, 1), blocked=blocked)[(0, 1, 2)] == 1
    assert (0, 1, 1) not in walk.global_distances((0, 1, 2), blocked=blocked)


def test_a_blocked_edge_shows_up_in_the_stacked_fields():
    walk = WalkMap({0: {(0, 0): set(), (0, 1): set(), (1, 0): set(), (1, 1): set()}})
    grids = walk.fields([0], (0, 0, 0), (4, 4), unknown=99.0, fill=0.0,
                        blocked={((0, 0, 1), (0, 0, 0))})
    assert grids[0][0, 1] == 3.0


def test_a_door_walked_one_way_is_still_a_way_back(h5_path):
    """No episode ever left Oak's lab, so every recorded doorway points
    inwards. Without the reverse, no tile in Pallet has any cost from inside
    the lab at all: the player stepped out, back in through the same door, and
    spent 482 actions on a flat field."""
    path = write(h5_path, [[(0, 12, 11), (40, 11, 5), (40, 11, 6)]])
    walk = build(path)
    assert walk.doors[(0, 12, 11)] == {(40, 11, 5)}
    out = walk.global_distances((0, 12, 11), two_way_doors=True)
    assert out[(40, 11, 5)] == 1
    assert out[(40, 11, 6)] == 2
    # Off by default: it cost more than it bought. Giving the planner a door
    # as a shortcut between rooms it had never walked that way took two of
    # three seeds from four milestones to none.
    assert (40, 11, 5) not in walk.global_distances((0, 12, 11))
