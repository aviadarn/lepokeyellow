"""Game version registry: ROM identification and per-version WRAM addresses.

Gen-1 WRAM is not shared between versions. Yellow moved `wGBC` into HRAM, so
every WRAM symbol above $CF1A sits one byte lower than in Red/Blue. Reading a
Red address out of a Yellow ROM returns a plausible-looking byte, so the
mistake is silent: `party_count` would read `wPartySpecies` and report a
species index as a count.

Addresses therefore live here, keyed by a version detected from the ROM header
rather than passed in by hand. `confidence` records how each address was
obtained; anything marked "inferred" must be confirmed against a real ROM with
`scripts/calibrate_ram.py` before it is trusted.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

# Cartridge header fields, same layout on every Game Boy ROM.
TITLE_SLICE = slice(0x134, 0x144)
GLOBAL_CHECKSUM_SLICE = slice(0x14E, 0x150)


@dataclass(frozen=True)
class RamMap:
    """WRAM addresses for one game version."""

    party_count: int
    party_species: int
    cur_map: int
    y_coord: int
    x_coord: int
    in_battle: int
    confidence: dict[str, str] = field(default_factory=dict)

    def verified(self, name: str) -> bool:
        return self.confidence.get(name) != "inferred"


@dataclass(frozen=True)
class Game:
    key: str
    title: str
    header_titles: tuple[str, ...]
    ram: RamMap


# Red/Blue values match pret/pokered and the published Red/Blue RAM map, and
# are the ones stmonty/lePokeRed ships. Yellow's party block start (0xD162) is
# documented; cur_map is derived from the -1 shift rule and is flagged as such.
RED = Game(
    key="red",
    title="Pokemon Red/Blue",
    header_titles=("POKEMON RED", "POKEMON BLUE"),
    ram=RamMap(
        party_count=0xD163,
        party_species=0xD164,
        cur_map=0xD35E,
        y_coord=0xD361,
        x_coord=0xD362,
        in_battle=0xD057,
        confidence={
            "party_count": "documented",
            "party_species": "documented",
            "cur_map": "documented",
            "y_coord": "documented",
            "x_coord": "documented",
            "in_battle": "documented",
        },
    ),
)

YELLOW = Game(
    key="yellow",
    title="Pokemon Yellow",
    header_titles=("POKEMON YELLOW",),
    ram=RamMap(
        party_count=0xD162,
        party_species=0xD163,
        cur_map=0xD35D,
        y_coord=0xD360,
        x_coord=0xD361,
        in_battle=0xD056,
        confidence={
            # Watched go 0 -> 1 on this cartridge when Pikachu arrives.
            "party_count": "confirmed-on-rom",
            # Confirmed by running the cartridge, not by the shift rule alone:
            # cur_map reads 38 (REDS_HOUSE_2F) once the new game starts, y_coord
            # moves 6<->7 on down/up and x_coord 1<->4 on left/right. See
            # CONFIRMED_ON below.
            # Read 84 (0x54, Pikachu's Gen-1 internal index) in the dataset.
            "party_species": "confirmed-on-rom",
            "cur_map": "confirmed-on-rom",
            "y_coord": "confirmed-on-rom",
            "x_coord": "confirmed-on-rom",
            # Read 2 (trainer battle) during the forced rival fight in Oak's
            # lab while Red's 0xD057 stayed 0, and cleared to 0 when it ended.
            "in_battle": "confirmed-on-rom",
        },
    ),
)

# ROM the Yellow addresses above were confirmed against, so a later
# disagreement can be traced to a different dump rather than to the table.
CONFIRMED_ON = {
    "yellow": "cc7d03262ebfaf2f06772c1a480c7d9d5f4a38e1",  # Yellow (UE) [C][!]
}

GAMES: dict[str, Game] = {game.key: game for game in (RED, YELLOW)}

# Sentinel the game stores in an empty species slot.
EMPTY_SPECIES = 0xFF


def rom_header_title(rom_path: str | Path) -> str:
    """Return the cartridge title, NUL- and space-stripped, upper-cased."""
    with open(rom_path, "rb") as handle:
        handle.seek(TITLE_SLICE.start)
        raw = handle.read(TITLE_SLICE.stop - TITLE_SLICE.start)
    if len(raw) < TITLE_SLICE.stop - TITLE_SLICE.start:
        raise ValueError(f"{rom_path} is too small to be a Game Boy ROM")
    return raw.split(b"\x00")[0].decode("ascii", "replace").strip().upper()


def rom_global_checksum(rom_path: str | Path) -> int:
    with open(rom_path, "rb") as handle:
        handle.seek(GLOBAL_CHECKSUM_SLICE.start)
        raw = handle.read(2)
    return int.from_bytes(raw, "big")


def rom_sha1(rom_path: str | Path) -> str:
    digest = hashlib.sha1()
    with open(rom_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identify(rom_path: str | Path) -> Game:
    """Pick the Game whose header title matches this ROM.

    Raises rather than guessing: a wrong RAM map fails silently at runtime, so
    an unknown ROM has to be named explicitly through `resolve(game=...)`.
    """
    title = rom_header_title(rom_path)
    for game in GAMES.values():
        if any(title.startswith(prefix) for prefix in game.header_titles):
            return game
    known = ", ".join(sorted(GAMES))
    raise ValueError(
        f"unrecognised ROM header title {title!r} in {rom_path}. "
        f"Known versions: {known}. Pass --game explicitly to override."
    )


def resolve(rom_path: str | Path, game: str | None = None) -> Game:
    """Resolve the Game for a ROM, cross-checking an explicit override."""
    if game is None:
        return identify(rom_path)
    if game not in GAMES:
        raise ValueError(f"unknown game {game!r}; known: {', '.join(sorted(GAMES))}")
    chosen = GAMES[game]
    title = rom_header_title(rom_path)
    if not any(title.startswith(prefix) for prefix in chosen.header_titles):
        raise ValueError(
            f"--game {game} does not match ROM header title {title!r}. "
            "Refusing to read another version's RAM addresses."
        )
    return chosen
