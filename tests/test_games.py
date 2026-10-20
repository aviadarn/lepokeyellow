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
    for field in ("party_count", "party_species", "cur_map", "y_coord",
                  "x_coord", "in_battle"):
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


def test_confirmed_addresses_are_not_flagged_as_inferred():
    """cur_map, y_coord and x_coord started as shift-rule guesses and were then
    checked by running the cartridge: cur_map reads 38 on a new game, y_coord
    moves 6<->7 on down/up, x_coord 1<->4 on left/right."""
    for field in ("cur_map", "y_coord", "x_coord"):
        assert YELLOW.ram.confidence[field] == "confirmed-on-rom", field
        assert YELLOW.ram.verified(field), field


def test_in_battle_is_still_only_inferred():
    """It reads 0 out of battle, which is what a wrong address also reads. Not
    confirmed until something has been observed while a battle runs."""
    assert not YELLOW.ram.verified("in_battle")


def test_confirmed_addresses_name_the_rom_they_were_checked_against():
    """So a later disagreement points at a different dump, not at the table."""
    from poke.games import CONFIRMED_ON

    assert len(CONFIRMED_ON["yellow"]) == 40


def test_the_shift_rule_holds_for_every_address():
    for field in ("party_count", "party_species", "cur_map", "y_coord",
                  "x_coord", "in_battle"):
        assert getattr(RED.ram, field) - getattr(YELLOW.ram, field) == 1, field


def test_yellow_is_a_dual_mode_cart_so_the_env_must_pick_a_mode(yellow_rom, monkeypatch):
    """PyBoy boots a CGB-flagged cart in colour, where R, G and B differ. The
    encoder is Conv2d(1, ...) and every writer takes channel 0, so the env has
    to force DMG or it feeds the model an arbitrary channel."""
    import poke.env as env_module

    seen = {}

    class FakePyBoy:
        def __init__(self, rom, **kwargs):
            seen.update(kwargs)

        def stop(self):
            pass

    monkeypatch.setattr(env_module, "PyBoy", FakePyBoy)
    env_module.PokeEnv(str(yellow_rom), "unused.state")
    assert seen["cgb"] is False

    seen.clear()
    env_module.PokeEnv(str(yellow_rom), "unused.state", color=True)
    assert "cgb" not in seen
