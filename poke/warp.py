"""Go anywhere in the game, by letting the game take you there.

The emulator states stop at Viridian Forest, and a dataset meant to cover the
whole game cannot be collected from there. Writing the player's map and
coordinates into WRAM does not work: the view pointer, block coordinates,
NPC sprite state, tileset, CGB palettes and Pikachu's follow state would all
go stale. The game's own warp routine sets every one of them. So:

  * hook `WarpFound2` -- by then the game has read the destination of the
    warp the player stepped on into hWarpDestinationMap / wDestinationWarpID
    -- and overwrite both with the target;
  * walk the player onto any warp (the BFS oracle does the walking);
  * the game loads the target map and places the player at that warp.

Warp tables come out of the ROM through pret's map header pointers, so every
map id and every warp is known without a hand-written list. A warp into an
indoor map is fine as far as it goes, but its exits that lead "back outside"
(destination LAST_MAP) read wLastMap, which still names wherever the player
teleported from. `teleport` therefore sets wLastMap to the outdoor map that
really leads in, found by walking the warp graph backwards.

Training data only: this is emulator surgery, like the oracle itself.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass

from poke.games import PARTY_MON_BYTES
from poke.oracle import _has_control, search
from poke.ram import cur_map, party_count

# hWarpDestinationMap value meaning "the outdoor map you came in from".
LAST_MAP = 0xFF
# Tilesets the game treats as outside (CheckIfInOutsideMap).
OVERWORLD, PLATEAU = 0, 23
NUM_MAPS = 0xF8

# party_struct offsets (pret's party_struct macro).
OFF_HP, OFF_BOX_LEVEL, OFF_EXP = 1, 3, 14
OFF_LEVEL, OFF_MAX_HP, OFF_ATTACK = 33, 34, 36
STAT_OFFSETS = (36, 38, 40, 42)  # attack, defense, speed, special


@dataclass(frozen=True)
class Warp:
    y: int
    x: int
    dest_warp: int
    dest_map: int


@dataclass(frozen=True)
class MapInfo:
    map_id: int
    name: str
    tileset: int
    height: int
    width: int
    warps: tuple[Warp, ...]

    @property
    def outdoor(self) -> bool:
        return self.tileset in (OVERWORLD, PLATEAU)


def _offset(bank: int, address: int) -> int:
    return address if address < 0x4000 else bank * 0x4000 + (address - 0x4000)


def _word(rom: bytes, at: int) -> int:
    return rom[at] | rom[at + 1] << 8


def read_maps(rom: bytes, symbols: dict[str, tuple[int, int]]) -> dict[int, MapInfo]:
    """Every map whose header parses as a real map, keyed by map id.

    Header layout (pret map_header): tileset, height, width, blocks, text,
    script pointers, connection flags, 11 bytes per connection, object
    pointer. Object data: border block, warp count, then y, x, destination
    warp, destination map per warp. Unused map ids point at junk, so a header
    whose fields are out of range is skipped rather than trusted.
    """
    labels = {location: name[:-2] for name, location in symbols.items()
              if name.endswith("_h")}
    pointers = _offset(*symbols["MapHeaderPointers"])
    banks = _offset(*symbols["MapHeaderBanks"])
    maps: dict[int, MapInfo] = {}
    for map_id in range(NUM_MAPS):
        bank, address = rom[banks + map_id], _word(rom, pointers + 2 * map_id)
        if not 0x4000 <= address < 0x8000 or (bank, address) not in labels:
            continue
        header = _offset(bank, address)
        tileset, height, width = rom[header], rom[header + 1], rom[header + 2]
        connections = bin(rom[header + 9] & 0x0F).count("1")
        objects = _offset(bank, _word(rom, header + 10 + 11 * connections))
        count = rom[objects + 1]
        if tileset > PLATEAU or not (0 < height < 128 and 0 < width < 128) or count > 32:
            continue
        warps = tuple(Warp(*rom[objects + 2 + 4 * k: objects + 6 + 4 * k])
                      for k in range(count))
        maps[map_id] = MapInfo(map_id, labels[(bank, address)], tileset,
                               height, width, warps)
    return maps


def outdoor_parents(maps: dict[int, MapInfo]) -> dict[int, int]:
    """For each map, the outdoor map a player walks in from.

    Breadth-first from every outdoor map along warps, so an indoor map behind
    another indoor map (Mt. Moon B1F behind 1F) inherits the outdoor map at
    the start of the chain -- which is what wLastMap holds when a player gets
    there on foot.
    """
    parent: dict[int, int] = {}
    queue = collections.deque()
    for map_id, info in maps.items():
        if info.outdoor:
            parent[map_id] = map_id
            queue.append(map_id)
    while queue:
        here = queue.popleft()
        for warp in maps[here].warps:
            there = warp.dest_map
            if there in maps and there not in parent and not maps[there].outdoor:
                parent[there] = parent[here]
                queue.append(there)
    return parent


class Teleporter:
    def __init__(self, env, symbols: dict[str, tuple[int, int]], rom: bytes):
        self.env = env
        self.symbols = symbols
        self.maps = read_maps(rom, symbols)
        self.parents = outdoor_parents(self.maps)
        self._target: tuple[int, int] | None = None

    def _redirect(self, _context) -> None:
        target_map, target_warp = self._target
        memory = self.env.pyboy.memory
        memory[self.symbols["hWarpDestinationMap"][1]] = target_map
        memory[self.symbols["wDestinationWarpID"][1]] = target_warp

    def teleport(self, target_map: int, target_warp: int = 0,
                 max_nodes: int = 3000, settle_limit: int = 60) -> bool:
        """Walk onto the nearest warp and arrive at `target_warp` of `target_map`.

        True only if the player ends up on the target map with control (the
        two-button rollback probe), so a state stuck in a fade or a script is
        reported as a failure rather than returned.
        """
        info = self.maps[target_map]
        if not 0 <= target_warp < len(info.warps):
            raise ValueError(f"{info.name} has {len(info.warps)} warps, not {target_warp + 1}")
        env = self.env
        start = cur_map(env)
        self._target = (target_map, target_warp)
        bank, address = self.symbols["WarpFound2"]
        env.pyboy.hook_register(bank, address, self._redirect, None)
        try:
            result = search(env, lambda: cur_map(env) == target_map,
                            {start, target_map}, buttons=("up", "down", "left", "right"),
                            with_facing=False, max_nodes=max_nodes)
        finally:
            env.pyboy.hook_deregister(bank, address)
        if not result.found:
            return False
        if not info.outdoor and target_map in self.parents:
            env.pyboy.memory[self.symbols["wLastMap"][1]] = self.parents[target_map]
        for _ in range(settle_limit):
            if _has_control(env):
                return cur_map(env) == target_map
            env.settle(10)
        return False


def set_level(env, level: int, slot: int = 0) -> None:
    """Raise one party member to `level`, scaling its stats the way Gen 1 does.

    Gen-1 stats are (base-dependent term) * L / 100 + 5, and HP adds L + 10,
    so the level-dependent part scales linearly in L. Scaling the current
    numbers keeps the Pokemon's own base stats and DVs without reading the
    base-stat table. Experience is set to the medium-fast curve (L^3), which
    is Pikachu's: below its level's threshold the game would "level" it DOWN
    after the next battle.
    """
    if not 2 <= level <= 100:
        raise ValueError(level)
    if slot >= party_count(env):
        raise ValueError(f"no party member in slot {slot}")
    memory = env.pyboy.memory
    base = env.game.ram.party_hp - OFF_HP + slot * PARTY_MON_BYTES
    old = memory[base + OFF_LEVEL]
    if old == 0:
        # The party count flips before the struct is written: the colour
        # Pikachu state, saved on that flip, reads all zeros here.
        raise ValueError(f"party slot {slot} is not filled in yet")

    def word(at):
        return memory[base + at] << 8 | memory[base + at + 1]

    def put(at, value):
        value = max(1, min(999, int(round(value))))
        memory[base + at], memory[base + at + 1] = value >> 8, value & 0xFF

    put(OFF_MAX_HP, (word(OFF_MAX_HP) - old - 10) * level / old + level + 10)
    put(OFF_HP, word(OFF_MAX_HP))
    for at in STAT_OFFSETS:
        put(at, (word(at) - 5) * level / old + 5)
    exp = level ** 3
    memory[base + OFF_EXP], memory[base + OFF_EXP + 1], memory[base + OFF_EXP + 2] = (
        exp >> 16, exp >> 8 & 0xFF, exp & 0xFF)
    memory[base + OFF_LEVEL] = level
    memory[base + OFF_BOX_LEVEL] = level


# Gen-1 text encoding: A-Z are 0x80-0x99 and 0x50 ends a name.
NAME_LENGTH = 11
NAME_END = 0x50


def encode_name(text: str) -> bytes:
    if not text or len(text) >= NAME_LENGTH or not all("A" <= c <= "Z" for c in text):
        raise ValueError(f"name must be 1-10 capital letters, got {text!r}")
    body = bytes(0x80 + ord(c) - ord("A") for c in text) + bytes([NAME_END])
    return body.ljust(NAME_LENGTH, bytes([NAME_END]))


def set_names(env, symbols: dict[str, tuple[int, int]], player: str = "RED",
              rival: str = "BLUE", starter: str = "PIKACHU") -> None:
    """Rename player, rival and the first party member.

    Bootstrap clears the intro by pressing A, which names everybody
    "AAAAAAA...". Those names are on screen in every battle and most
    dialogue, and the longplay's player is RED with an un-nicknamed PIKACHU.
    The starter's OT name is rewritten with the player's: Yellow only lets
    Pikachu follow when its OT name and ID match the player's.
    """
    memory = env.pyboy.memory
    for symbol, text in (("wPlayerName", player), ("wRivalName", rival),
                         ("wPartyMonNicks", starter), ("wPartyMonOT", player)):
        if symbol in ("wPartyMonNicks", "wPartyMonOT") and party_count(env) == 0:
            continue
        address = symbols[symbol][1]
        for offset, value in enumerate(encode_name(text)):
            memory[address + offset] = value


#: wOptions low nibble: letter delay in frames. 1 FAST, 3 MEDIUM (the
#: default, and what every bootstrapped state has), 5 SLOW.
TEXT_FAST, TEXT_MEDIUM, TEXT_SLOW = 1, 3, 5


def set_text_speed(env, symbols: dict[str, tuple[int, int]], speed: int) -> None:
    """The longplay's player set TEXT SPEED: FAST on the options screen before
    starting (video frame ~3420). Text cadence is what the model reads A
    presses from, so synthetic data for it has to print at the same speed."""
    address = symbols["wOptions"][1]
    memory = env.pyboy.memory
    memory[address] = (memory[address] & 0xF0) | speed
