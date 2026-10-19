import pytest

from poke.games import EMPTY_SPECIES, RED, YELLOW
from poke.ram import cur_map, has_pokemon, party_count, party_species
from tests.conftest import FakeEnv


def test_reads_through_the_games_ram_map():
    """Same byte value at each version's own address, read correctly by both."""
    for game in (RED, YELLOW):
        env = FakeEnv(game, {game.ram.party_count: 3})
        assert party_count(env) == 3


def test_red_address_read_on_yellow_would_be_wrong():
    """Yellow's party_species sits where Red keeps party_count.

    This is the silent failure the registry prevents: a Red-addressed read of
    a Yellow game returns the species index and calls it a count.
    """
    env = FakeEnv(YELLOW, {YELLOW.ram.party_count: 1, YELLOW.ram.party_species: 25})
    assert party_count(env) == 1
    assert env.read_ram(RED.ram.party_count) == 25


def test_missing_address_reads_as_zero():
    assert party_count(FakeEnv(YELLOW)) == 0


def test_empty_species_sentinel_is_none():
    env = FakeEnv(YELLOW, {YELLOW.ram.party_species: EMPTY_SPECIES})
    assert party_species(env) is None


def test_has_pokemon_boundaries():
    env = FakeEnv(YELLOW, {YELLOW.ram.party_count: 0})
    assert not has_pokemon(env)
    for count in (1, 6):
        env.write(YELLOW.ram.party_count, count)
        assert has_pokemon(env)


def test_impossible_party_count_raises_instead_of_returning_true():
    """A count above 6 means the address is wrong, not that the task is solved.
    Treating it as success would silently pass every experiment."""
    env = FakeEnv(YELLOW, {YELLOW.ram.party_count: 200})
    with pytest.raises(RuntimeError, match="address is wrong"):
        has_pokemon(env)


def test_cur_map_passthrough():
    env = FakeEnv(YELLOW, {YELLOW.ram.cur_map: 40})
    assert cur_map(env) == 40
    assert cur_map(FakeEnv(YELLOW)) is None
