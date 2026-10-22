import pytest

from poke.flags import (
    GOT_OAKS_PARCEL, GOT_POKEDEX, OAKS_PARCEL, OAK_GOT_PARCEL,
    bag_items, carrying_parcel, event_bit,
)
from poke.games import RED, YELLOW
from tests.conftest import FakeEnv


def env_with(flags=(), bag=(), game=YELLOW):
    """FakeEnv whose event bytes carry `flags` and whose bag carries `bag`."""
    ram = {}
    for index in flags:
        addr = game.ram.event_flags + index // 8
        ram[addr] = ram.get(addr, 0) | (1 << (index % 8))
    ram[game.ram.num_bag_items] = len(bag)
    for slot, (item, qty) in enumerate(bag):
        ram[game.ram.num_bag_items + 1 + 2 * slot] = item
        ram[game.ram.num_bag_items + 2 + 2 * slot] = qty
    return FakeEnv(game, ram)


def test_event_bit_reads_the_right_byte_and_bit():
    env = env_with(flags=[GOT_POKEDEX])
    assert event_bit(env, GOT_POKEDEX) == 1
    assert event_bit(env, GOT_OAKS_PARCEL) == 0
    # Bit 37 is byte 4, bit 5 -- 0xD746 + 4 == 0xD74A, as measured.
    assert YELLOW.ram.event_flags + GOT_POKEDEX // 8 == 0xD74A
    assert YELLOW.ram.event_flags + OAK_GOT_PARCEL // 8 == 0xD74D


def test_unreadable_ram_is_zero_not_a_crash():
    assert event_bit(FakeEnv(YELLOW, {}), GOT_POKEDEX) == 0


def test_bag_items_stop_at_the_count():
    env = env_with(bag=[(OAKS_PARCEL, 1), (0x14, 3)])
    assert bag_items(env) == [(OAKS_PARCEL, 1), (0x14, 3)]


def test_absurd_bag_count_reads_as_empty_rather_than_walking_wram():
    env = FakeEnv(YELLOW, {YELLOW.ram.num_bag_items: 200})
    assert bag_items(env) == []


def test_carrying_is_the_bag_not_the_flag():
    """Bit 57 never clears on delivery, so a flag-based 'carrying' would be
    true forever and the delivery objective would never end. Measured on the
    cartridge: after Oak takes it, bit 57 stays 1 and the bag is empty."""
    delivered = env_with(flags=[GOT_OAKS_PARCEL, OAK_GOT_PARCEL], bag=[])
    assert event_bit(delivered, GOT_OAKS_PARCEL) == 1
    assert not carrying_parcel(delivered)
    assert carrying_parcel(env_with(flags=[GOT_OAKS_PARCEL], bag=[(OAKS_PARCEL, 1)]))


def test_yellow_event_flags_are_red_minus_one_and_facing_is_not():
    """Everything above $CF1A shifts by one in Yellow; facing is below it."""
    assert YELLOW.ram.event_flags == RED.ram.event_flags - 1
    assert YELLOW.ram.num_bag_items == RED.ram.num_bag_items - 1
    assert YELLOW.ram.facing == RED.ram.facing == 0xC109
    assert YELLOW.ram.facing < 0xCF1A < YELLOW.ram.event_flags


def test_every_new_yellow_address_declares_its_confidence():
    for name in ("event_flags", "num_bag_items", "facing"):
        assert name in YELLOW.ram.confidence


def test_reading_yellow_event_indices_out_of_red_is_refused():
    """games.py keeps a Yellow ROM from being read with Red addresses. The
    bit *indices* here are Yellow's too, and they were being applied to
    whatever game the env resolved -- the same silent-plausible-byte failure
    one level up."""
    with pytest.raises(RuntimeError, match="event bit indices"):
        event_bit(env_with(game=RED), GOT_POKEDEX)
    with pytest.raises(RuntimeError, match="event bit indices"):
        carrying_parcel(env_with(game=RED))
