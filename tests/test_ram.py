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


def test_battle_kind_distinguishes_wild_from_trainer():
    """Gen 1 stores 1 for a wild battle and 2 for a trainer; the rival fight in
    Oak's lab reads 2."""
    from poke.ram import NO_BATTLE, TRAINER_BATTLE, WILD_BATTLE, battle_kind, in_battle

    env = FakeEnv(YELLOW, {YELLOW.ram.in_battle: NO_BATTLE})
    assert battle_kind(env) == NO_BATTLE and not in_battle(env)
    for kind in (WILD_BATTLE, TRAINER_BATTLE):
        env.write(YELLOW.ram.in_battle, kind)
        assert battle_kind(env) == kind and in_battle(env)


def test_unreadable_battle_flag_is_not_a_battle():
    from poke.ram import battle_kind, in_battle

    env = FakeEnv(YELLOW)
    assert battle_kind(env) == 0 and not in_battle(env)
