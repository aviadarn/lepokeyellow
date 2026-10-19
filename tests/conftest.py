import numpy as np
import pytest

from poke.games import GLOBAL_CHECKSUM_SLICE, TITLE_SLICE


def make_rom(tmp_path, title: str, name: str = "fake.gb", size: int = 1 << 16):
    """A file with a valid Game Boy header title and nothing else.

    Enough to exercise identification without going near a real ROM.
    """
    data = bytearray(size)
    encoded = title.encode("ascii")
    assert len(encoded) <= TITLE_SLICE.stop - TITLE_SLICE.start
    data[TITLE_SLICE.start : TITLE_SLICE.start + len(encoded)] = encoded
    data[GLOBAL_CHECKSUM_SLICE] = (0x1234).to_bytes(2, "big")
    path = tmp_path / name
    path.write_bytes(bytes(data))
    return path


@pytest.fixture
def yellow_rom(tmp_path):
    return make_rom(tmp_path, "POKEMON YELLOW", "pokeyellow.gb")


@pytest.fixture
def red_rom(tmp_path):
    return make_rom(tmp_path, "POKEMON RED", "pokered.gb")


class FakeEnv:
    """Stands in for PokeEnv: a RAM dict plus the resolved game."""

    def __init__(self, game, ram=None):
        self.game = game
        self.ram = dict(ram or {})

    def read_ram(self, addr):
        return self.ram.get(addr)

    def write(self, addr, value):
        self.ram[addr] = value

    def screen(self):
        return np.zeros((144, 160, 3), dtype=np.uint8)
