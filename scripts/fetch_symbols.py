"""Download pret/pokeyellow's symbol file to data/sym/ (see poke/symbols.py)."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

from poke.symbols import DEFAULT_PATH, SYMBOLS_URL, load


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_PATH))
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(SYMBOLS_URL, timeout=60) as response:
        out.write_bytes(response.read())
    print(f"wrote {out}: {len(load(out))} symbols")


if __name__ == "__main__":
    main()
