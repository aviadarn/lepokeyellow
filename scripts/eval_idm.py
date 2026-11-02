"""Replay closure: do the IDM's buttons reproduce the episode they were read from?

Accuracy on held bits cannot say whether labels are good enough to train on:
most frames' bits are invisible either way, and one wrong frame in a menu
can send the game somewhere else. Replay can. For each held-out synthetic
episode (map-disjoint, the split train_idm.py uses):

  * replay the TRUE buttons from the episode's start state, snapshotting
    every SEGMENT frames -- and check this reproduces the recorded frames
    exactly, or nothing below means anything;
  * for every segment, restore its true start state, press the IDM's
    buttons (held_p > 0.5) for SEGMENT frames, and count the frames until
    the screen first differs from the recording.

Reported beside two floors: all buttons released, and the episode's own
buttons shifted by one frame (how much does timing alone cost?).

  uv run python -m scripts.eval_idm --checkpoint checkpoints/idm/best.pt
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import torch

from poke.synth import Pad, clean_index
from scripts.label_video import decode, load_model, predict
from scripts.train_idm import memmaps, split

ROM = "pokeyellow.gbc"
SEGMENT = 300


def start_env(state: str):
    from poke.env import PokeEnv
    from poke.symbols import load
    from poke.warp import TEXT_FAST, set_names, set_text_speed

    env = PokeEnv(ROM, state, color=True)
    env.reset()
    symbols = load()
    set_names(env, symbols)
    set_text_speed(env, symbols, TEXT_FAST)
    return env


def context_table(truth: np.ndarray, pred: np.ndarray, matter: np.ndarray,
                  in_battle: np.ndarray, dialog: np.ndarray) -> dict[str, dict[str, list[int]]]:
    """Per situation and button, on the sampled frames where a press could
    show: [caught, missed, wrongly pressed, correctly released]."""
    from poke.synth import BUTTONS

    out: dict[str, dict[str, list[int]]] = {}
    for t in np.flatnonzero(matter[:, 0] != 255):
        where = "battle" if in_battle[t] else "text box" if dialog[t] else "overworld"
        for k, name in enumerate(BUTTONS):
            if not matter[t, k]:
                continue
            row = out.setdefault(where, {}).setdefault(name, [0, 0, 0, 0])
            pressed, predicted = truth[t] >> k & 1, pred[t] >> k & 1
            row[(0 if predicted else 1) if pressed else (2 if predicted else 3)] += 1
    return out


def _add(total: dict, part: dict) -> None:
    for where, buttons in part.items():
        for name, row in buttons.items():
            acc = total.setdefault(where, {}).setdefault(name, [0, 0, 0, 0])
            for i, v in enumerate(row):
                acc[i] += v


def tiles(sequence) -> list[tuple]:
    """Distinct consecutive (map, y, x): the path, with the timing removed."""
    out = []
    for tile in sequence:
        if not out or out[-1] != tile:
            out.append(tile)
    return out


def closure(env, clean, masks, candidates: dict[str, np.ndarray], colours,
            truth_tiles=None) -> dict:
    """Per candidate and SEGMENT-frame segment: frames until the first pixel
    difference (strict), and whether the replay walks the same path of tiles
    and ends on the same tile as the recording (timing-free)."""
    pyboy = env.pyboy
    pad = Pad(pyboy)
    snapshots = []
    exact = True
    for t, mask in enumerate(masks):
        if t % SEGMENT == 0:
            buffer = io.BytesIO()
            pyboy.save_state(buffer)
            snapshots.append((t, buffer.getvalue(), int(masks[t - 1]) if t else 0))
        pad.set(int(mask))
        pyboy.tick(1, True)
        if t % 25 == 0 and not np.array_equal(
                clean_index(np.asarray(pyboy.screen.ndarray[:, :, :3])[None], colours)[0], clean[t]):
            exact = False
    out = {"truth_replay_exact": exact, "segments": len(snapshots)}
    from poke.ram import coords, cur_map

    for name, predicted in candidates.items():
        survived, same_path, same_end = [], [], []
        for start, blob, before in snapshots:
            stop = min(start + SEGMENT, len(masks))
            pyboy.load_state(io.BytesIO(blob))
            pad.reset()
            pad.set(before)
            frames = None
            walked = []
            for t in range(start, stop):
                pad.set(int(predicted[t]))
                pyboy.tick(1, True)
                walked.append((cur_map(env), *coords(env)))
                if frames is None:
                    screen = clean_index(np.asarray(pyboy.screen.ndarray[:, :, :3])[None],
                                         colours)[0]
                    if not np.array_equal(screen, clean[t]):
                        frames = t - start
            survived.append((stop - start if frames is None else frames, stop - start))
            if truth_tiles is not None:
                truth = [tuple(int(v) for v in row) for row in truth_tiles[start:stop]]
                same_path.append(tiles(walked) == tiles(truth))
                same_end.append(walked[-1] == truth[-1])
        survived = np.array(survived)
        out[name] = {"segments_reproduced": float((survived[:, 0] == survived[:, 1]).mean()),
                     "median_frames_to_divergence": float(np.median(survived[:, 0])),
                     "frames": survived[:, 0].tolist()}
        if same_path:
            out[name]["same_path"] = float(np.mean(same_path))
            out[name]["same_end_tile"] = float(np.mean(same_end))
    pad.reset()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="checkpoints/idm/best.pt")
    parser.add_argument("--synth", default="data/synth")
    parser.add_argument("--val-share", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--unseen-from", type=int, default=None,
                        help="evaluate episodes numbered >= this instead of the map "
                             "split (for a model trained while collection ran)")
    parser.add_argument("--min-episode", type=int, default=None,
                        help="keep only held-out episodes numbered >= this (a batch, e.g. "
                             "20000 for the bump-A episodes)")
    parser.add_argument("--by-context", action="store_true",
                        help="also count caught / missed presses per situation (overworld, text, battle)")
    parser.add_argument("--out", default="results/idm_closure.json")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, checkpoint = load_model(args.checkpoint, device)
    colours = np.load("data/colours_cgb.npy")
    episodes = memmaps(Path(args.synth))
    if args.unseen_from is None:
        _, val, held_out = split(episodes, args.val_share, args.seed)
    else:
        val = [e for e in episodes if int(e["stem"].split("_")[1]) >= args.unseen_from]
        held_out = f"episodes >= {args.unseen_from}"
    if args.min_episode is not None:
        val = [e for e in val if int(e["stem"].split("_")[1]) >= args.min_episode]
    rows, by_context = [], {}
    for episode in val[:args.episodes]:
        frames = np.asarray(episode["frames"])
        with np.load(Path(args.synth) / f"{episode['stem']}.npz") as data:
            clean = data["clean"]
            truth_tiles = np.stack([data["map_ids"], data["ys"], data["xs"]], 1)
            situation = (data["in_battles"], data["dialog_opens"])
        masks = np.asarray(episode["buttons"]).astype(np.int64)
        held, _, _, onset = predict(model, lambda index: frames[index], len(frames), device)
        onset = onset if checkpoint["has_onset"] else None
        candidates = {"idm": decode(held, onset),
                      "idm_overlapping_dirs": decode(held, onset, single_direction=False),
                      "idm_held_only": decode(held, None),
                      "released": np.zeros_like(masks),
                      "truth_late_1": np.concatenate([[0], masks[:-1]])}
        if args.by_context:
            _add(by_context, context_table(masks, candidates["idm"], np.asarray(episode["mattered"]),
                                           *situation))
        env = start_env(episode["state"])
        try:
            row = closure(env, clean, masks, candidates, colours, truth_tiles)
        finally:
            env.close()
        row["state"] = episode["state"]
        rows.append(row)
        print(f"{Path(episode['state']).stem:28s} exact={row['truth_replay_exact']} "
              + "  ".join(f"{k}: px {row[k]['segments_reproduced']:.2f} "
                          f"path {row[k]['same_path']:.2f} end {row[k]['same_end_tile']:.2f}"
                          for k in candidates), flush=True)
    summary = {k: {"segments_reproduced": float(np.mean([r[k]["segments_reproduced"] for r in rows])),
                   "median_frames_to_divergence": float(np.median(
                       [f for r in rows for f in r[k]["frames"]])),
                   "same_path": float(np.mean([r[k]["same_path"] for r in rows])),
                   "same_end_tile": float(np.mean([r[k]["same_end_tile"] for r in rows]))}
               for k in ("idm", "idm_overlapping_dirs", "idm_held_only", "released",
                         "truth_late_1")}
    summary["truth_replay_exact"] = all(r["truth_replay_exact"] for r in rows)
    if args.by_context:
        summary["by_context"] = by_context
        for where, buttons in sorted(by_context.items()):
            for name, (hit, miss, wrong, ok) in sorted(buttons.items()):
                if hit + miss:
                    print(f"{where:10s} {name:6s} caught {hit}/{hit + miss}  "
                          f"wrongly pressed {wrong}/{wrong + ok}", flush=True)
    print(json.dumps(summary), flush=True)
    Path(args.out).write_text(json.dumps(
        {"checkpoint": args.checkpoint, "step": int(checkpoint["step"]),
         "held_out_maps": held_out, "summary": summary, "episodes": rows}, indent=1))


if __name__ == "__main__":
    main()
