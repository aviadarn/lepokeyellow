import pytest

from poke.games import YELLOW
from poke.warp import (LAST_MAP, MapInfo, Warp, encode_name, outdoor_parents,
                       read_maps, set_level)


def test_encode_name_uses_gen1_letters_and_terminator():
    assert encode_name("RED") == bytes([0x91, 0x84, 0x83]) + bytes([0x50] * 8)
    for bad in ("", "red", "ABCDEFGHIJK", "R3D"):
        with pytest.raises(ValueError):
            encode_name(bad)


def info(map_id, tileset, warps=()):
    return MapInfo(map_id, f"M{map_id}", tileset, 9, 10, tuple(warps))


def test_parents_follow_warps_back_to_the_outdoor_map():
    maps = {
        0: info(0, 0, [Warp(5, 5, 0, 37)]),               # town -> house
        37: info(37, 1, [Warp(7, 2, 0, LAST_MAP), Warp(1, 7, 0, 38)]),
        38: info(38, 4, [Warp(1, 7, 1, 37)]),            # upstairs
        99: info(99, 2),                                 # unreachable
    }
    parents = outdoor_parents(maps)
    assert parents == {0: 0, 37: 0, 38: 0}


def test_read_maps_parses_a_header_and_its_warps():
    rom = bytearray(0x4000 * 64)
    pointers, banks, header, objects = 0x3F * 0x4000, 0x3F * 0x4000 + 0x200, 0x4000, 0x4100
    rom[pointers:pointers + 2] = (0x4000).to_bytes(2, "little")    # map 0 -> 01:4000
    rom[banks] = 1
    rom[header:header + 3] = bytes([0, 9, 10])                      # tileset, h, w
    rom[header + 9] = 0b0001                                        # one connection
    rom[header + 10 + 11:header + 10 + 13] = (0x4100).to_bytes(2, "little")
    rom[objects:objects + 2] = bytes([0x0B, 2])                     # border, 2 warps
    rom[objects + 2:objects + 10] = bytes([5, 5, 0, 37, 11, 12, 1, 40])
    symbols = {"MapHeaderPointers": (0x3F, 0x4000), "MapHeaderBanks": (0x3F, 0x4200),
               "Town_h": (1, 0x4000)}
    maps = read_maps(bytes(rom), symbols)
    assert list(maps) == [0]
    town = maps[0]
    assert (town.name, town.tileset, town.height, town.width) == ("Town", 0, 9, 10)
    assert town.warps == (Warp(5, 5, 0, 37), Warp(11, 12, 1, 40)) and town.outdoor


class Memory(bytearray):
    pass


class MemEnv:
    """A party of one L5 Pikachu, laid out as the cartridge had it."""

    game = YELLOW

    def __init__(self):
        self.pyboy = type("P", (), {})()
        self.pyboy.memory = Memory(0x10000)
        base = YELLOW.ram.party_hp - 1
        # Bytes read off the colour Pikachu state once the struct was filled.
        struct = bytes.fromhex("54001300001717a3542d0000a84a00007d0000000000000000"
                               "000093821e280000050013000b0008000e000a")
        self.pyboy.memory[base:base + 44] = struct
        self.pyboy.memory[YELLOW.ram.party_count] = 1

    def read_ram(self, addr):
        return self.pyboy.memory[addr]


def word(env, at):
    return env.pyboy.memory[at] << 8 | env.pyboy.memory[at + 1]


def test_set_level_scales_like_gen1_and_matches_the_cartridge():
    env = MemEnv()
    set_level(env, 15)
    base = YELLOW.ram.party_hp - 1
    # The game showed 37/37 for this Pikachu at L15 after this exact edit.
    assert word(env, base + 34) == 37 and word(env, base + 1) == 37
    assert env.pyboy.memory[base + 33] == 15 and env.pyboy.memory[base + 3] == 15
    assert word(env, base + 36) == (11 - 5) * 3 + 5          # attack
    exp = int.from_bytes(env.pyboy.memory[base + 14:base + 17], "big")
    assert exp == 15 ** 3


def test_set_level_refuses_an_unfilled_party_slot():
    env = MemEnv()
    env.pyboy.memory[YELLOW.ram.party_hp - 1 + 33] = 0
    with pytest.raises(ValueError, match="not filled"):
        set_level(env, 10)


def test_set_text_speed_keeps_the_other_option_bits():
    from poke.warp import TEXT_FAST, set_text_speed

    env = MemEnv()
    address = 0xD354
    env.pyboy.memory[address] = 0b1100_0011          # anim off, SET, MEDIUM
    set_text_speed(env, {"wOptions": (0, address)}, TEXT_FAST)
    assert env.pyboy.memory[address] == 0b1100_0001
