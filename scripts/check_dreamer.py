"""Gate A for the Yellow PokeDreamer set: does it load, replay and cover?

  1. PokeDreamer's own README loader runs on every chunk, and every chunk
     matches poke.dreamer.SCHEMA.
  2. Replay: for sampled episodes, pressing the recorded actions from the
     recorded start state reproduces every recorded frame and probe exactly.
     This is the guarantee the dataset makes; a policy that rolled the
     emulator back would break it.
  3. Coverage beside PokeDreamer Red (two of its chunks, downloaded to
     data/pokedreamer_red/), per 10 k steps: distinct maps, distinct
     (map, y, x) tiles, map changes, battle and dialog shares, action mix.

  uv run python -m scripts.check_dreamer
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from poke.dreamer import POKEDREAMER_KEYS, SCHEMA, act, check, probe

ROM = "pokeyellow.gbc"
REPLAY_KEYS = ("obs", "map_ids", "xs", "ys", "facings", "in_battles", "dialog_opens",
               "badges", "party_hps", "party_max_hps", "party_levels")


def coverage(chunk) -> dict:
    """Per-chunk coverage numbers, from PokeDreamer's own arrays only."""
    maps = chunk["map_ids"]
    tiles = np.stack([maps, chunk["ys"], chunk["xs"]], 1)
    changes = int((maps[1:] != maps[:-1]).sum())
    return {
        "steps": int(len(maps)),
        "maps": int(len(np.unique(maps))),
        "tiles": int(len(np.unique(tiles, axis=0))),
        "map_changes": changes,
        "battle_share": round(float(chunk["in_battles"].mean()), 3),
        "dialog_share": round(float(chunk["dialog_opens"].mean()), 3),
        "action_share": np.round(np.bincount(chunk["actions"], minlength=8)
                                 / len(maps), 3).tolist(),
    }


def replay(episode_file: Path, state: str) -> dict:
    from poke.env import PokeEnv
    from poke.symbols import load
    from poke.warp import set_names

    with np.load(episode_file) as data:
        recorded = {key: data[key] for key in REPLAY_KEYS + ("actions",)}
    env = PokeEnv(ROM, state, color=True)
    try:
        env.reset()
        set_names(env, load())
        mismatches: dict[str, int] = {}
        first = None
        for t, action in enumerate(recorded["actions"]):
            fields = probe(env)
            for key in REPLAY_KEYS:
                if not np.array_equal(np.asarray(fields[key]), recorded[key][t]):
                    mismatches[key] = mismatches.get(key, 0) + 1
                    first = t if first is None else first
            act(env, int(action))
    finally:
        env.close()
    return {"file": str(episode_file), "steps": int(len(recorded["actions"])),
            "exact": not mismatches, "mismatches": mismatches, "first_mismatch": first}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/pokedreamer_yellow")
    parser.add_argument("--red", default="data/pokedreamer_red")
    parser.add_argument("--replays", type=int, default=12)
    parser.add_argument("--out", default="results/dreamer_gate_a.json")
    args = parser.parse_args()

    data = Path(args.data)
    chunks = sorted(data.glob("transitions_*.npz"))
    report: dict = {"chunks": len(chunks)}

    # 1. The README loader, then the full schema.
    yellow_rows, total = [], 0
    all_maps, all_tiles = set(), set()
    for path in chunks:
        chunk = np.load(path)
        frames, actions = chunk["obs"], chunk["actions"]          # README, verbatim
        assert frames.shape[1:] == (144, 160, 3) and len(actions) == len(frames)
        check({key: chunk[key] for key in SCHEMA})
        yellow_rows.append(coverage(chunk))
        total += len(actions)
        all_maps.update(np.unique(chunk["map_ids"]).tolist())
        all_tiles.update(map(tuple, np.unique(np.stack(
            [chunk["map_ids"], chunk["ys"], chunk["xs"]], 1), axis=0).tolist()))
    report["loader"] = {"ok": True, "steps": total,
                        "pokedreamer_keys_present": list(POKEDREAMER_KEYS)}
    print(f"1. loader + schema ok on {len(chunks)} chunks, {total} steps", flush=True)

    # 2. Replay a spread of episodes, both policies.
    meta = json.loads((data / "collection.json").read_text())
    episodes = meta["episodes"]
    picks = np.linspace(0, len(episodes) - 1, min(args.replays, len(episodes))).astype(int)
    replays = [replay(Path(episodes[k]["file"]), episodes[k]["state"]) for k in picks]
    exact = sum(row["exact"] for row in replays)
    report["replay"] = {"exact": exact, "of": len(replays), "rows": replays}
    print(f"2. replay exact on {exact}/{len(replays)} episodes", flush=True)
    for row in replays:
        if not row["exact"]:
            print(f"   {row['file']}: first mismatch at {row['first_mismatch']}, "
                  f"{row['mismatches']}", flush=True)

    # 3. Coverage per 10 k steps, ours against PokeDreamer Red.
    red_rows = [coverage(np.load(path)) for path in sorted(Path(args.red).glob("*.npz"))]

    def mean(rows, key):
        return round(float(np.mean([row[key] for row in rows])), 3) if rows else None

    table = {name: {key: mean(rows, key) for key in
                    ("maps", "tiles", "map_changes", "battle_share", "dialog_share")}
             for name, rows in (("yellow", yellow_rows), ("red_pokedreamer", red_rows))}
    table["yellow"]["whole_set_maps"] = len(all_maps)
    table["yellow"]["whole_set_tiles"] = len(all_tiles)
    report["coverage_per_10k"] = table
    report["red_chunks"] = red_rows
    print("3. per 10 k steps (mean over chunks):", flush=True)
    for name, row in table.items():
        print(f"   {name:16s} {row}", flush=True)

    Path(args.out).write_text(json.dumps(report, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
