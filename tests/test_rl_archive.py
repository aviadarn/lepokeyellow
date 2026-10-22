"""The archive hands out training start states. Training only: it restores."""

import random

from poke.flags import BATTLED_RIVAL_IN_OAKS_LAB, OAK_APPEARED_IN_PALLET
from poke.games import YELLOW
from poke.rl.archive import CELL, Archive


class FakeEnv:
    def __init__(self, tile=(12, 35, 10), bits=(OAK_APPEARED_IN_PALLET, BATTLED_RIVAL_IN_OAKS_LAB),
                 white=0.0, battle=0):
        self.game = YELLOW
        self.tile = tile
        self.white = white
        self.battle = battle
        self.ram_values = {}
        for bit in bits:
            addr = YELLOW.ram.event_flags + bit // 8
            self.ram_values[addr] = self.ram_values.get(addr, 0) | (1 << (bit % 8))
        self.snapshots = 0

    def read_ram(self, addr):
        if addr == YELLOW.ram.cur_map:
            return self.tile[0]
        if addr == YELLOW.ram.y_coord:
            return self.tile[1]
        if addr == YELLOW.ram.x_coord:
            return self.tile[2]
        if addr == YELLOW.ram.in_battle:
            return self.battle
        if addr == YELLOW.ram.num_bag_items:
            return 0
        return self.ram_values.get(addr, 0)

    def screen(self):
        import numpy as np
        shade = 255 if self.white else 0
        frame = np.zeros((144, 160, 3), dtype=np.uint8)
        frame[96:144, :, :] = shade
        return frame

    def snapshot(self):
        # Keyed by tile: two FakeEnvs are separate objects, so a per-instance
        # counter would hand both cells the same blob and the sampling test
        # could not tell them apart.
        self.snapshots += 1
        return f"state-{self.tile}".encode()


def test_a_cell_is_objective_map_and_a_coarse_tile():
    env = FakeEnv(tile=(12, 35, 10))
    archive = Archive()
    assert archive.key_for(env) == (2, 12, 35 // CELL, 10 // CELL)


def test_adjacent_tiles_share_a_cell():
    archive = Archive()
    near = archive.key_for(FakeEnv(tile=(12, 35, 10)))
    also = archive.key_for(FakeEnv(tile=(12, 35, 11)))
    assert near == also


def test_adding_a_new_cell_reports_new_and_adding_it_again_does_not():
    archive, env = Archive(), FakeEnv()
    assert archive.add(env) is True
    assert archive.add(env) is False
    assert len(archive) == 1


def test_archive_refuses_a_state_without_control():
    """Review focus 3: a snapshot taken during a dialogue starts an episode
    the policy cannot act in."""
    archive = Archive()
    assert archive.add(FakeEnv(white=1.0)) is False
    assert archive.add(FakeEnv(battle=1)) is False
    assert len(archive) == 0


def test_sample_returns_a_stored_blob():
    archive, env = Archive(rng=random.Random(0)), FakeEnv()
    archive.add(env)
    assert archive.sample() == b"state-(12, 35, 10)"


def test_sample_prefers_the_cell_with_fewer_visits():
    archive = Archive(rng=random.Random(0))
    archive.add(FakeEnv(tile=(12, 0, 0)))
    archive.add(FakeEnv(tile=(12, 40, 40)))
    busy = archive.key_for(FakeEnv(tile=(12, 0, 0)))
    for _ in range(400):
        archive.note_visit(busy)
    picks = [archive.sample() for _ in range(200)]
    quiet = picks.count(b"state-(12, 40, 40)")
    assert quiet > 150, f"the quiet cell was picked {quiet}/200 times"


def test_an_empty_archive_samples_none():
    assert Archive().sample() is None


def test_capacity_evicts_the_most_visited_cell():
    archive = Archive(capacity=2, rng=random.Random(0))
    archive.add(FakeEnv(tile=(12, 0, 0)))
    archive.add(FakeEnv(tile=(12, 40, 40)))
    busy = archive.key_for(FakeEnv(tile=(12, 0, 0)))
    for _ in range(10):
        archive.note_visit(busy)
    archive.add(FakeEnv(tile=(1, 20, 20)))
    assert len(archive) == 2
    assert busy not in archive.cells


def test_stats_report_cells_maps_and_bytes():
    archive = Archive()
    archive.add(FakeEnv(tile=(12, 0, 0)))
    archive.add(FakeEnv(tile=(1, 20, 20)))
    stats = archive.stats()
    assert stats["cells"] == 2
    assert stats["maps"] == [1, 12]
    assert stats["bytes"] == sum(len(c.blob) for c in archive.cells.values())
