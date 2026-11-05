"""Write a party and a bag into the game, as a screen showed them.

A window of the longplay replayed from a door's save state carries that
state's party and bag. What the video's own party menu and item list show --
species or nickname, level, HP, status; the field moves a member's submenu
lists; item names and quantities -- is written as shown.
What no screen shows is guessed and returned, so a report can say which:
DVs (8 each), stat experience (0), the four moves a species starts with,
and the stats those give.
"""

from __future__ import annotations

from poke.screen_text import CHARS
from poke.screens import FIELD_MOVES
from poke.warp import _offset

MON_BYTES = 44
NAME_BYTES = 11
SPECIES_COUNT = 190
MOVE_COUNT = 165
STATUS = {"SLP": 0x03, "PSN": 0x08, "BRN": 0x10, "FRZ": 0x20, "PAR": 0x40}
DV = 8


def _decode(raw: bytes) -> str:
    return "".join(CHARS.get(c, "#") for c in raw.split(b"\x50", 1)[0])


def _encode(text: str, size: int = NAME_BYTES) -> bytes:
    codes = {ch: code for code, ch in CHARS.items()}
    body = bytes(codes[ch] for ch in text) + b"\x50"
    return body.ljust(size, b"\x50")[:size]


def species_by_name(rom: bytes, symbols) -> dict[str, int]:
    """Species name -> internal species id (1-190), from MonsterNames."""
    start = _offset(*symbols["MonsterNames"])
    names = {}
    for i in range(SPECIES_COUNT):
        name = _decode(rom[start + 10 * i:start + 10 * i + 10])
        if name and "#" not in name and name not in names:
            names[name] = i + 1
    return names


def _names(rom: bytes, start: int, count: int) -> dict[str, int]:
    """Name -> id (from 1) for `count` names one after another."""
    at, out = start, {}
    for i in range(1, count + 1):
        end = rom.index(b"\x50", at)
        out.setdefault(_decode(rom[at:end]), i)
        at = end + 1
    return out


def items_by_name(rom: bytes, symbols) -> dict[str, int]:
    """Item name -> item id, from ItemNames."""
    return _names(rom, _offset(*symbols["ItemNames"]), 97)


def moves_by_name(rom: bytes, symbols) -> dict[str, int]:
    """Move name -> move id, from MoveNames."""
    return _names(rom, _offset(*symbols["MoveNames"]), MOVE_COUNT)


def _pp(rom: bytes, symbols, move: int) -> int:
    return rom[_offset(*symbols["Moves"]) + (move - 1) * 6 + 5] if move else 0


def _base(rom: bytes, symbols, species: int) -> bytes:
    dex = rom[_offset(*symbols["PokedexOrder"]) + species - 1]
    start = _offset(*symbols["BaseStats"]) + (dex - 1) * 28
    return rom[start:start + 28]


def _exp(growth: int, level: int) -> int:
    """Gen 1's experience for a level, per growth rate."""
    n = level
    return {0: n ** 3, 3: max(0, 6 * n ** 3 // 5 - 15 * n * n + 100 * n - 140),
            4: 4 * n ** 3 // 5, 5: 5 * n ** 3 // 4}.get(growth, n ** 3)


def set_party(pyboy, rom: bytes, symbols, party: list[dict]) -> dict:
    """Replace the party with `party` ([{name, level, hp, max_hp, status}],
    as `poke.screens.party_screen` reads it). Names that are species names
    set the species; any other is taken as a nickname of the slot's current
    species. Returns {field: what was guessed}."""
    m = pyboy.memory
    species_ids = species_by_name(rom, symbols)
    count, species_at = symbols["wPartyCount"][1], symbols["wPartySpecies"][1]
    mon_at, ot_at = symbols["wPartyMon1"][1], symbols["wPartyMonOT"][1]
    nick_at = symbols["wPartyMonNicks"][1]
    player = bytes(m[symbols["wPlayerName"][1]:symbols["wPlayerName"][1] + NAME_BYTES])
    player_id = bytes(m[symbols["wPlayerID"][1]:symbols["wPlayerID"][1] + 2])
    for slot, mon in enumerate(party):
        species = species_ids.get(mon["name"]) or m[mon_at + slot * MON_BYTES] or species_ids["PIKACHU"]
        base = _base(rom, symbols, species)
        level = mon["level"]
        struct = bytearray(MON_BYTES)
        struct[0] = species
        struct[1:3] = mon["hp"].to_bytes(2, "big")
        struct[3] = level
        struct[4] = STATUS.get(mon["status"] or "", 0)
        struct[5], struct[6], struct[7] = base[6], base[7], base[8]
        moves = [mv for mv in base[15:19]]
        struct[8:12] = bytes(moves)
        struct[12:14] = player_id
        struct[14:17] = _exp(base[19], level).to_bytes(3, "big")
        struct[27:29] = bytes([DV << 4 | DV, DV << 4 | DV])
        struct[29:33] = bytes(_pp(rom, symbols, mv) for mv in moves)
        struct[33] = level
        struct[34:36] = mon["max_hp"].to_bytes(2, "big")
        for k, stat in enumerate(base[2:6]):
            value = (stat + DV) * 2 * level // 100 + 5
            struct[36 + 2 * k:38 + 2 * k] = value.to_bytes(2, "big")
        for i, byte in enumerate(struct):
            m[mon_at + slot * MON_BYTES + i] = byte
        for i, byte in enumerate(_encode(mon["name"])):
            m[nick_at + slot * NAME_BYTES + i] = byte
        for i, byte in enumerate(player):
            m[ot_at + slot * NAME_BYTES + i] = byte
        m[species_at + slot] = species
    m[count] = len(party)
    m[species_at + len(party)] = 0xFF
    return {"moves": "the species' starting moves", "dvs": DV, "stat_exp": 0,
            "stats": "from base stats, DV 8", "experience": "the level's threshold"}


def set_field_moves(pyboy, rom: bytes, symbols, slot: int, names: list[str]) -> None:
    """Give party member `slot` the field moves its submenu listed, in that
    order, in its first move slots, with full PP. Its other moves fill the
    rest, less any field move the submenu did not list."""
    m = pyboy.memory
    ids = moves_by_name(rom, symbols)
    field = {ids[n] for n in FIELD_MOVES}
    at = symbols["wPartyMon1"][1] + slot * MON_BYTES
    rest = [mv for mv in m[at + 8:at + 12] if mv and mv not in field]
    moves = ([ids[n] for n in names] + rest + [0] * 4)[:4]
    for i, mv in enumerate(moves):
        m[at + 8 + i], m[at + 29 + i] = mv, _pp(rom, symbols, mv)


def set_bag(pyboy, rom: bytes, symbols, items: list[tuple[str, int]]) -> None:
    """Put `items` ((name, quantity), as `poke.screens.bag_screen` reads
    them) first in the bag, in order, then whatever else it held."""
    m = pyboy.memory
    ids = items_by_name(rom, symbols)
    count, at = symbols["wNumBagItems"][1], symbols["wBagItems"][1]
    held = [(m[at + 2 * i], m[at + 2 * i + 1]) for i in range(m[count])]
    shown = [(ids[n], q) for n, q in items if n in ids]
    rest = [(item, q) for item, q in held if item not in {i for i, _ in shown}]
    bag = (shown + rest)[:20]
    for i, (item, q) in enumerate(bag):
        m[at + 2 * i], m[at + 2 * i + 1] = item, q
    m[at + 2 * len(bag)] = 0xFF
    m[count] = len(bag)
