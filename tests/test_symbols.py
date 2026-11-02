"""RamMap against pret's symbol table, when it has been fetched."""

import pytest

from poke.games import PARTY_MON_BYTES, YELLOW
from poke.symbols import DEFAULT_PATH, constants, load

SYMBOL_FOR = {
    "party_count": "wPartyCount",
    "party_species": "wPartySpecies",
    "cur_map": "wCurMap",
    "y_coord": "wYCoord",
    "x_coord": "wXCoord",
    "in_battle": "wIsInBattle",
    "event_flags": "wEventFlags",
    "num_bag_items": "wNumBagItems",
    "facing": "wSpritePlayerStateData1FacingDirection",
    "badges": "wObtainedBadges",
    "party_hp": "wPartyMon1HP",
    "party_max_hp": "wPartyMon1MaxHP",
    "party_level": "wPartyMon1Level",
    "map_scripts": "wGameProgressFlags",
}

needs_symbols = pytest.mark.skipif(
    not DEFAULT_PATH.exists(), reason="run scripts/fetch_symbols.py first")


def test_load_parses_bank_and_address(tmp_path):
    sym = tmp_path / "x.sym"
    sym.write_text("; comment\n00:d35d wCurMap\n03:402d _Joypad ; trailing\n"
                   "01 BIKESHOP_CLERK\n")
    assert load(sym) == {"wCurMap": (0, 0xD35D), "_Joypad": (3, 0x402D)}
    assert constants(sym) == {"BIKESHOP_CLERK": 1}


@needs_symbols
@pytest.mark.parametrize("field", sorted(SYMBOL_FOR))
def test_yellow_ram_map_agrees_with_pret(field):
    table = load()
    assert table[SYMBOL_FOR[field]] == (0, getattr(YELLOW.ram, field)), field


@needs_symbols
def test_party_stride_matches_pret():
    table = load()
    assert table["wPartyMon2"][1] - table["wPartyMon1"][1] == PARTY_MON_BYTES


@needs_symbols
def test_tile_map_matches_pret():
    from poke.ram import TILE_MAP

    assert load()["wTileMap"] == (0, TILE_MAP)


@needs_symbols
def test_progress_block_length_matches_pret():
    from poke.games import NUM_PROGRESS_BYTES

    table = load()
    assert (table["wGameProgressFlagsEnd"][1] - table["wGameProgressFlags"][1]
            == NUM_PROGRESS_BYTES)
