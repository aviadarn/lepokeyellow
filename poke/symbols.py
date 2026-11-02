"""pret/pokeyellow's symbol table: names for the cartridge's RAM and routines.

pret builds a byte-identical Yellow ROM from source and publishes the linker's
symbol file on the repository's `symbols` branch. It is not vendored (no
licence is stated on it); `scripts/fetch_symbols.py` downloads it to
data/sym/. Two uses:

  * cross-checking RamMap: every Yellow address confirmed on the cartridge
    must agree with it, which is what makes its other addresses worth using;
  * routine addresses for emulator hooks, e.g. WarpFound2 for teleporting.
"""

from __future__ import annotations

from pathlib import Path

SYMBOLS_URL = ("https://raw.githubusercontent.com/pret/pokeyellow/symbols/"
               "pokeyellow.sym")
DEFAULT_PATH = Path("data/sym/pokeyellow.sym")


def _entries(path):
    for line in Path(path).read_text().splitlines():
        line = line.split(";", 1)[0].strip()
        if line:
            yield line.split(maxsplit=1)


def load(path: str | Path = DEFAULT_PATH) -> dict[str, tuple[int, int]]:
    """{name: (bank, address)} from an rgbds .sym file.

    Exported constants (`01 BIKESHOP_CLERK`, no bank) are not locations and
    are left to `constants`.
    """
    table: dict[str, tuple[int, int]] = {}
    for location, name in _entries(path):
        if ":" in location:
            bank, address = location.split(":")
            table[name] = (int(bank, 16), int(address, 16))
    return table


def constants(path: str | Path = DEFAULT_PATH) -> dict[str, int]:
    """{name: value} for the exported constants in a .sym file."""
    return {name: int(value, 16) for value, name in _entries(path)
            if ":" not in value}
