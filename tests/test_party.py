from pathlib import Path

import numpy as np
import pytest

ROM = Path("pokeyellow.gbc")
STATE = Path("states/cgb/teleport/PalletTown_w0.state")
COLOURS = Path("data/colours_cgb.npy")
pytestmark = pytest.mark.skipif(not (ROM.exists() and STATE.exists() and COLOURS.exists()),
                                reason="needs the ROM, a colour state and the colour table")


def _press(pyboy, name, wait):
    pyboy.button_press(name)
    pyboy.tick(4, True)
    pyboy.button_release(name)
    pyboy.tick(wait, True)


def test_a_written_party_shows_in_the_party_menu():
    from poke.env import PokeEnv
    from poke.party import set_party
    from poke.screen_text import font
    from poke.screens import party_screen
    from poke.symbols import load
    from poke.video import Snapper
    from scripts.sync_opening import emulator_index, text_classes

    rom, symbols = ROM.read_bytes(), load()
    env = PokeEnv(str(ROM), str(STATE), color=True)
    env.reset()
    party = [{"name": "WARTORTLE", "level": 26, "hp": 24, "max_hp": 74, "status": None},
             {"name": "PIDGEOTTO", "level": 27, "hp": 77, "max_hp": 77, "status": "PAR"},
             {"name": "PIKACHU", "level": 26, "hp": 62, "max_hp": 62, "status": "SLP"}]
    guessed = set_party(env.pyboy, rom, symbols, party)
    assert "moves" in guessed and "dvs" in guessed
    env.pyboy.memory[symbols["wBattleAndStartSavedMenuItem"][1]] = 1      # POKéMON
    _press(env.pyboy, "start", 40)
    _press(env.pyboy, "a", 90)
    snapper = Snapper(np.load(COLOURS))
    frame = emulator_index(env.pyboy, snapper)
    shown = party_screen(frame, text_classes(snapper.colours), font(rom, symbols))
    assert shown == party


def test_a_written_bag_lists_its_items():
    from poke.env import PokeEnv
    from poke.party import items_by_name, set_bag
    from poke.symbols import load

    rom, symbols = ROM.read_bytes(), load()
    env = PokeEnv(str(ROM), str(STATE), color=True)
    env.reset()
    set_bag(env.pyboy, rom, symbols, [("BICYCLE", 1), ("POTION", 3)])
    m, at = env.pyboy.memory, symbols["wBagItems"][1]
    ids = items_by_name(rom, symbols)
    assert m[symbols["wNumBagItems"][1]] >= 2
    assert [m[at + i] for i in range(4)] == [ids["BICYCLE"], 1, ids["POTION"], 3]


def test_written_field_moves_show_in_the_party_submenu():
    from poke.env import PokeEnv
    from poke.party import set_field_moves, set_party
    from poke.screen_text import font
    from poke.screens import party_submenu
    from poke.symbols import load
    from poke.video import Snapper
    from scripts.sync_opening import emulator_index, text_classes

    rom, symbols = ROM.read_bytes(), load()
    env = PokeEnv(str(ROM), str(STATE), color=True)
    env.reset()
    set_party(env.pyboy, rom, symbols,
              [{"name": "PIKACHU", "level": 26, "hp": 62, "max_hp": 62, "status": None},
               {"name": "IVYSAUR", "level": 25, "hp": 70, "max_hp": 70, "status": None}])
    set_field_moves(env.pyboy, rom, symbols, 1, ["CUT"])
    env.pyboy.memory[symbols["wBattleAndStartSavedMenuItem"][1]] = 1      # POKéMON
    _press(env.pyboy, "start", 40)
    _press(env.pyboy, "a", 90)
    _press(env.pyboy, "down", 20)
    _press(env.pyboy, "a", 40)
    snapper = Snapper(np.load(COLOURS))
    frame = emulator_index(env.pyboy, snapper)
    assert party_submenu(frame, text_classes(snapper.colours), font(rom, symbols)) == (1, "IVYSAUR", ["CUT"])
