"""Per-frame synthetic episodes for the inverse dynamics model (poke/synth.py).

Each episode: a start state from the colour pool, renamed to the longplay's
cast and set to TEXT SPEED: FAST as the longplay's player chose, driven a frame at a time by the human-like controller; every frame's
8 held buttons recorded; one frame in `--sample-every` gets the exact
`mattered` mask; then the whole episode goes through the video's path
(colour correction, 5x, x264 CRF 28, snap) so the model sees what the video
shows. Clean colour indices are kept beside the degraded ones.

  uv run python -m scripts.synth_collect --episodes 320 --frames 3000 --workers 8

Writes data/synth/ep_XXXXX.npz:
  frames (T,144,160) u1 degraded colour index    clean (T,144,160) u1
  buttons (T,) u1 (bit k = poke.synth.BUTTONS[k])
  mattered (T,8) u1: 0/1 on sampled frames, 255 elsewhere
  map_ids ys xs (T,) u1, in_battles dialog_opens (T,) bool, snap_mean (T,) f2
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROM = "pokeyellow.gbc"
BOOT = "boot"
#: Intro episodes are numbered from here, apart from the state-pool ones.
INTRO_BASE = 10_000


def _episode(job):
    k, state, frames, sample_every, seed, out, options = job
    from poke.env import PokeEnv
    from poke.symbols import load
    from poke.synth import (BUTTONS, HumanLike, MenuTapper, boost_encounters, clean_index, degrade,
                            mattered, play)
    from poke.video import Snapper
    from poke.warp import TEXT_FAST, set_names, set_text_speed

    colours = np.load("data/colours_cgb.npy")
    rng = np.random.default_rng(seed)
    started = time.time()
    if state == BOOT:
        # Power-on, blank cartridge RAM: the intro, title, menus and naming
        # screens, tapped through by MenuTapper.
        env = PokeEnv(ROM, None, color=True)
    else:
        env = PokeEnv(ROM, state, color=True)
    try:
        if state == BOOT:
            rgb, masks, labels, blobs = play(env, frames, rng, sample_every,
                                             policy=MenuTapper(rng, idle=options["idle"]))
        else:
            env.reset()
            symbols = load()
            set_names(env, symbols)
            set_text_speed(env, symbols, TEXT_FAST)
            boost = (lambda e: boost_encounters(e, symbols)) if options.get("encounters") else None
            rgb, masks, labels, blobs = play(
                env, frames, rng, sample_every, on_map_change=boost,
                policy=HumanLike(rng, bump_a=options["bump_a"],
                                 text_hold=options.get("text_hold", 0.0)))
        played = time.time() - started
        matter = np.full((frames, len(BUTTONS)), 255, np.uint8)
        for t, blob in blobs.items():
            if t + 32 <= frames:
                before = int(masks[t - 1]) if t else 0
                matter[t] = mattered(env, blob, masks[t:], before)
    finally:
        env.close()
    countered = time.time() - started - played
    snapper = Snapper(colours)
    degraded, snap_mean = degrade(rgb, snapper)
    clean = clean_index(rgb, colours)
    path = Path(out) / f"ep_{k:05d}.npz"
    np.savez_compressed(path, frames=degraded, clean=clean, buttons=masks, mattered=matter,
                        snap_mean=snap_mean, **labels)
    sampled = matter[matter[:, 0] != 255]
    return {"episode": k, "state": state, "frames": frames, "file": str(path),
            "unknown_colour_px": int((clean == 255).sum()),
            "pixel_error": float((degraded != clean).mean()),
            "mattered_share": sampled.mean(0).round(3).tolist() if len(sampled) else None,
            "seconds": {"play": round(played, 1), "mattered": round(countered, 1),
                        "total": round(time.time() - started, 1)}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=320)
    parser.add_argument("--frames", type=int, default=3000)
    parser.add_argument("--sample-every", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/synth")
    parser.add_argument("--manifest", default="results/start_states_cgb.json")
    parser.add_argument("--skip-existing", action="store_true",
                        help="only episodes whose file is missing (resume a run)")
    parser.add_argument("--intro", type=int, default=0,
                        help="ALSO this many power-on episodes (ep_1XXXX), and only those "
                             "if --episodes is 0")
    parser.add_argument("--first", type=int, default=0,
                        help="number of the first state-pool episode (a new batch beside "
                             "existing ones, e.g. 20000)")
    parser.add_argument("--intro-first", type=int, default=0,
                        help="power-on episodes are numbered from 10000 + this")
    parser.add_argument("--intro-idle", type=int, nargs=2, default=(0, 120),
                        help="frames before the first tap of a power-on episode, drawn "
                             "uniformly from [LO, HI); ~(1500, 3600) lets the intro play out")
    parser.add_argument("--bump-a", type=float, default=0.0,
                        help="chance of pressing A after a walk that went nowhere")
    parser.add_argument("--text-hold", type=float, default=0.0,
                        help="chance an A in a text box is a 10-120 frame hold of A or B")
    parser.add_argument("--encounters", action="store_true",
                        help="wild encounters on every grass step (wGrassRate = 255)")
    parser.add_argument("--tag", default=None,
                        help="write collection_TAG.json (train_idm reads every collection*.json)")
    args = parser.parse_args()
    options = {"idle": tuple(args.intro_idle), "bump_a": args.bump_a,
               "text_hold": args.text_hold, "encounters": args.encounters}

    from scripts.collect_dreamer import start_states

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    states = start_states(Path(args.manifest))
    numbers = range(args.first, args.first + args.episodes)
    intro_numbers = range(INTRO_BASE + args.intro_first, INTRO_BASE + args.intro_first + args.intro)
    jobs = [(k, states[k % len(states)], args.frames, args.sample_every,
             args.seed * 1_000_003 + k, str(out), options) for k in numbers
            if not (args.skip_existing and (out / f"ep_{k:05d}.npz").exists())]
    jobs += [(k, BOOT, args.frames, args.sample_every,
              args.seed * 1_000_003 + k, str(out), options) for k in intro_numbers
             if not (args.skip_existing and (out / f"ep_{k:05d}.npz").exists())]
    print(f"{len(jobs)} episodes x {args.frames} frames, {len(states)} start states",
          flush=True)
    rows, started = [], time.time()
    with Pool(args.workers) as pool:
        for row in pool.imap_unordered(_episode, jobs):
            rows.append(row)
            print(f"ep {row['episode']:4d} px_err {row['pixel_error']:.4f} "
                  f"unk {row['unknown_colour_px']:6d} {row['seconds']}  "
                  f"{Path(row['state']).stem}", flush=True)
    if args.skip_existing:
        # Rows for episodes an earlier run wrote are rebuilt from their files.
        done = {row["episode"] for row in rows}
        for k in numbers:
            path = out / f"ep_{k:05d}.npz"
            if k not in done and path.exists():
                rows.append({"episode": k, "state": states[k % len(states)],
                             "frames": args.frames, "file": str(path)})
        for k in intro_numbers:
            path = out / f"ep_{k:05d}.npz"
            if k not in done and path.exists():
                rows.append({"episode": k, "state": BOOT, "frames": args.frames,
                             "file": str(path)})
    rows.sort(key=lambda row: row["episode"])
    if args.tag:
        meta_name = f"collection_{args.tag}.json"
    else:
        meta_name = "collection.json" if args.episodes else "collection_intro.json"
    (out / meta_name).write_text(json.dumps(
        {"states": states, "frames": args.frames, "sample_every": args.sample_every,
         "seed": args.seed, "options": options, "seconds": round(time.time() - started, 1),
         "episodes": rows}, indent=1))
    print(f"{len(rows)} episodes in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
