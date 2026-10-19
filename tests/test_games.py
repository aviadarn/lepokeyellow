import pytest

from poke.games import GAMES, RED, YELLOW, identify, resolve, rom_header_title
from tests.conftest import make_rom


def test_identifies_yellow(yellow_rom):
    assert identify(yellow_rom) is YELLOW


def test_identifies_red_and_blue(tmp_path, red_rom):
    assert identify(red_rom) is RED
    assert identify(make_rom(tmp_path, "POKEMON BLUE", "blue.gb")) is RED


def test_unknown_rom_raises_rather_than_guessing(tmp_path):
    rom = make_rom(tmp_path, "SUPER MARIO LAND", "mario.gb")
    with pytest.raises(ValueError, match="unrecognised ROM header"):
        identify(rom)


def test_title_is_nul_stripped(yellow_rom):
    assert rom_header_title(yellow_rom) == "POKEMON YELLOW"


def test_truncated_file_raises(tmp_path):
    rom = tmp_path / "tiny.gb"
    rom.write_bytes(b"\x00" * 16)
    with pytest.raises(ValueError, match="too small"):
        rom_header_title(rom)


def test_explicit_game_must_match_the_rom(yellow_rom):
    """The whole point of the registry: refuse Red addresses on a Yellow ROM."""
    assert resolve(yellow_rom, "yellow") is YELLOW
    with pytest.raises(ValueError, match="Refusing to read another version"):
        resolve(yellow_rom, "red")


def test_unknown_game_override_raises(yellow_rom):
    with pytest.raises(ValueError, match="unknown game"):
        resolve(yellow_rom, "crystal")


def test_yellow_wram_sits_one_below_red():
    """Yellow moved wGBC to HRAM, shifting WRAM above $CF1A down a byte."""
    for field in ("party_count", "party_species", "cur_map"):
        red = getattr(RED.ram, field)
        yellow = getattr(YELLOW.ram, field)
        assert red - yellow == 1, field
        assert yellow > 0xCF1A, field


def test_red_addresses_match_upstream():
    """Keep Red numbers identical to lePokeRed so results stay comparable."""
    assert RED.ram.party_count == 0xD163
    assert RED.ram.party_species == 0xD164


def test_every_address_declares_its_provenance():
    for game in GAMES.values():
        for field in ("party_count", "party_species", "cur_map"):
            assert field in game.ram.confidence, (game.key, field)


def test_inferred_addresses_are_flagged_as_unverified():
    """cur_map for Yellow was derived from the shift rule, not read off a
    disassembly. calibrate_ram.py exists because of this."""
    assert not YELLOW.ram.verified("cur_map")
    assert YELLOW.ram.verified("party_count")
