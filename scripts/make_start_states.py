"""Colour start states across the whole game, for dataset collection.

TRAINING ONLY: emulator surgery. From one story state (default: just after
Oak hands over the Pokedex), for every map the warp graph can lead back out
of, rename the cast to the longplay's (RED / BLUE / PIKACHU), raise Pikachu to
a level that fits the area, and teleport there through the game's own warp
routine (poke/warp.py). A state is kept only if the player arrives on the
target map WITH control.

Writes states/cgb/teleport/<Map>_w<warp>.state and a manifest JSON listing
every attempt, kept or not, so a missing map is a recorded failure rather
than a silent gap.

  uv run python -m scripts.make_start_states --workers 8
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

ROM = "pokeyellow.gbc"

#: Pikachu's level by the outdoor area a map belongs to: roughly where a
#: player who is not grinding hard would be, so wild battles are winnable
#: without being one-sided. Keyed by pret map label.
AREA_LEVEL = {
    "PalletTown": 6, "Route1": 6, "ViridianCity": 8, "Route2": 9, "Route22": 9,
    "PewterCity": 12, "Route3": 14, "Route4": 16, "CeruleanCity": 20,
    "Route24": 20, "Route25": 21, "Route5": 22, "Route6": 22,
    "VermilionCity": 24, "Route11": 25, "Route9": 26, "Route10": 28,
    "LavenderTown": 30, "Route8": 32, "Route7": 32, "CeladonCity": 34,
    "Route16": 36, "Route12": 36, "Route17": 38, "Route18": 38, "Route13": 38,
    "Route14": 38, "Route15": 40, "FuchsiaCity": 40, "SaffronCity": 44,
    "Route19": 44, "Route20": 46, "Route21": 46, "CinnabarIsland": 48,
    "Route23": 55, "IndigoPlateau": 58,
}
DEFAULT_LEVEL = 30

#: Maps never used as start points: link-cable rooms, the credits room, and
#: the Pokemon League chambers whose entry scripts lock the doors behind you.
SKIP = {"TradeCenter", "Colosseum", "HallOfFame", "LoreleisRoom", "BrunosRoom",
        "AgathasRoom", "LancesRoom", "ChampionsRoom"}


def plan(maps, parents) -> list[tuple[int, int, int, str]]:
    """(map id, warp, level, label) for every teleport worth attempting.

    Warp 0 everywhere; on outdoor maps with three or more warps, the middle
    one as well, so a town is not always entered at the same door.
    """
    jobs = []
    for map_id, info in sorted(maps.items()):
        if info.name in SKIP or map_id not in parents or not info.warps:
            continue
        level = AREA_LEVEL.get(maps[parents[map_id]].name, DEFAULT_LEVEL)
        warps = [0]
        if info.outdoor and len(info.warps) >= 3:
            warps.append(len(info.warps) // 2)
        for warp in warps:
            jobs.append((map_id, warp, level, f"{info.name}_w{warp}"))
    return jobs


def _attempt(job):
    base, out_dir, (map_id, warp, level, label) = job
    from poke.env import PokeEnv
    from poke.ram import coords, cur_map
    from poke.symbols import load
    from poke.warp import Teleporter, set_level, set_names

    rom = Path(ROM).read_bytes()
    symbols = load()
    env = PokeEnv(ROM, base, color=True)
    started = time.time()
    try:
        env.reset()
        set_names(env, symbols)
        set_level(env, level)
        ok = Teleporter(env, symbols, rom).teleport(map_id, warp)
        y, x = coords(env)
        row = {"label": label, "map": map_id, "warp": warp, "level": level,
               "ok": ok, "arrived": [cur_map(env), y, x],
               "seconds": round(time.time() - started, 1)}
        if ok:
            path = Path(out_dir) / f"{label}.state"
            path.write_bytes(env.snapshot())
            row["state"] = str(path)
        return row
    except Exception as error:  # a failure here is a manifest row, not a crash
        return {"label": label, "map": map_id, "warp": warp, "level": level,
                "ok": False, "error": repr(error)}
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="states/cgb/oracle/pokedex.state")
    parser.add_argument("--out-dir", default="states/cgb/teleport")
    parser.add_argument("--manifest", default="results/start_states_cgb.json")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--only", nargs="+", help="map labels, e.g. PewterGym")
    args = parser.parse_args()

    from poke.symbols import load
    from poke.warp import outdoor_parents, read_maps

    maps = read_maps(Path(ROM).read_bytes(), load())
    jobs = plan(maps, outdoor_parents(maps))
    if args.only:
        jobs = [job for job in jobs if job[3].rsplit("_w", 1)[0] in args.only]
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} teleports from {args.base}", flush=True)

    rows = []
    with Pool(args.workers) as pool:
        for row in pool.imap_unordered(_attempt, [(args.base, args.out_dir, j) for j in jobs]):
            rows.append(row)
            status = "ok " if row["ok"] else "FAIL"
            print(f"{status} {row['label']:32s} {row.get('arrived', row.get('error'))}",
                  flush=True)
    rows.sort(key=lambda row: (row["map"], row["warp"]))
    kept = sum(row["ok"] for row in rows)
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(json.dumps(
        {"base": args.base, "attempted": len(rows), "kept": kept, "rows": rows}, indent=1))
    print(f"kept {kept}/{len(rows)}; wrote {args.manifest}")


if __name__ == "__main__":
    main()
