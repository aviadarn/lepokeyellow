"""Check the longplay's labels against what its screen proves, frame by frame.

Emulator replay (repair_labels.py) only reaches the opening. Past it, two
things in the picture itself say a button was pressed (poke/label_rules.py):
every step the player takes scrolls the screen with its direction held just
before, and every `▼` prompt that moves on was answered with a new A or B.
This script finds both across the whole video and reports how many of them
the labels explain.

  * Walking is reported only. On synthetic data the labels already explain
    98 % of steps and the ones they miss are forced movement, where adding
    a press was always wrong (results/label_rules_synth.json).
  * The text rule is applied with --apply, unless its inserted presses
    disagree with the emulator-verified labels of the opening (`verified`,
    `buttons_repaired` in --truth) more than --min-precision allows.

Writes results/label_consistency_<name>.json; with --apply also
`buttons_ruled`, `rule_changed` and `rule_event` (frames that are a step
start or a `▼` answer) into the labels file. Screen events are cached.

  uv run python -m scripts.check_labels --name v3 --apply
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np

from poke.label_rules import (answers, enforce_text, prompt_visible, scroll, step_starts,
                              text_agreement, text_change, walking_agreement)
from scripts.label_video import frames_for

#: An inserted tap is right if the verified labels press A or B this close.
MATCH_FRAMES = 8


def screen_events(video, chunk: int = 4096, limit: int | None = None):
    """(scroll (n,) int8, prompt (n,) bool, change (n,) float32) for every
    video frame, read `chunk` frames at a time with one frame of overlap."""
    refs = video["frame_ref"][:]
    n = len(refs) if limit is None else min(limit, len(refs))
    colours = video["colours"][:]
    directions = np.full(n, -1, np.int8)
    prompt = np.zeros(n, bool)
    change = np.zeros(n, np.float32)
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        first = max(lo - 1, 0)
        frames = frames_for(video, refs[first:hi])
        skip = lo - first
        directions[lo:hi] = scroll(frames)[skip:]
        prompt[lo:hi] = prompt_visible(frames, colours)[skip:]
        change[lo:hi] = text_change(frames)[skip:]
    return directions, prompt, change


def _onsets(masks: np.ndarray, bits=(4, 5)) -> np.ndarray:
    out = []
    for bit in bits:
        held = (masks >> bit) & 1
        out.append(np.flatnonzero(held & ~np.r_[0, held[:-1]]))
    return np.sort(np.concatenate(out))


def verified_precision(before: np.ndarray, after: np.ndarray, spans: np.ndarray,
                       truth: np.ndarray, verified: np.ndarray) -> tuple[float | None, int]:
    """Share of the A taps the rule added inside verified waits that land
    within MATCH_FRAMES of an A or B press in the verified labels."""
    added = (after >> 4) & 1 & ~((before >> 4) & 1)
    starts = np.flatnonzero(added & ~np.r_[0, added[:-1]])
    inside = [t for t in starts if any(s <= t <= e and verified[s:e + 1].all() for s, e in spans)]
    if not inside:
        return None, 0
    truth_onsets = _onsets(truth)
    hits = sum(bool(len(truth_onsets)) and np.abs(truth_onsets - t).min() <= MATCH_FRAMES for t in inside)
    return hits / len(inside), len(inside)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", default="data/longplay/video.h5")
    parser.add_argument("--labels", default="data/longplay/labels.h5")
    parser.add_argument("--truth", default=None,
                        help="labels file holding `verified` and `buttons_repaired` (default: --labels)")
    parser.add_argument("--name", default="labels")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--min-precision", type=float, default=0.9)
    parser.add_argument("--cache", default="data/longplay/screen_events.npz")
    parser.add_argument("--limit", type=int, default=None, help="only the first N frames")
    args = parser.parse_args()

    started = time.time()
    with h5py.File(args.video) as video:
        sha1 = str(video.attrs.get("video_sha1", ""))
        cache = Path(args.cache)
        cached = np.load(cache) if cache.exists() else None
        if cached is not None and str(cached["sha1"]) == sha1 and \
                (args.limit is None and int(cached["n"]) == len(video["frame_ref"])):
            directions, prompt, change = cached["scroll"], cached["prompt"], cached["change"]
        else:
            directions, prompt, change = screen_events(video, limit=args.limit)
            if args.limit is None:
                np.savez(cache, scroll=directions, prompt=prompt, change=change,
                         sha1=sha1, n=len(directions))
    n = len(directions)
    with h5py.File(args.labels) as labels:
        masks = labels["buttons"][:n].astype(np.int64)
        onset_a = labels["onset_p"][:n, 4].astype(np.float32)
        bad = labels["state"][:n].astype(np.int64)
        bad = (bad == 255).all(axis=1)
        if "cut" in labels:
            bad |= labels["cut"][:n].astype(bool)
    with h5py.File(args.truth or args.labels) as truth_file:
        has_truth = "verified" in truth_file and "buttons_repaired" in truth_file
        verified = truth_file["verified"][:n] if has_truth else np.zeros(n, bool)
        truth = truth_file["buttons_repaired"][:n].astype(np.int64) if has_truth else masks

    directions = directions.copy()
    directions[bad] = -1
    starts = step_starts(directions)
    spans = answers(prompt, change)
    bad_cum = np.r_[0, np.cumsum(bad)]
    spans = np.array([(s, e) for s, e in spans if bad_cum[e + 1] == bad_cum[s]], np.int64).reshape(-1, 2)

    ruled, changed = enforce_text(masks, spans, onset_a)
    precision, checked = verified_precision(masks, ruled, spans, truth, verified)
    apply_text = precision is None or precision >= args.min_precision
    if not apply_text:
        ruled, changed = masks.copy(), np.zeros(n, bool)

    report = {
        "name": args.name, "labels": args.labels, "frames": int(n),
        "steps": int(len(starts)), "walking_agreement": round(walking_agreement(masks, starts), 4),
        "answers": int(len(spans)),
        "text_agreement_before": round(text_agreement(masks, spans), 4),
        "text_agreement_after": round(text_agreement(ruled, spans), 4),
        "text_taps_added": int(len(np.flatnonzero(((ruled >> 4) & 1) & ~((masks >> 4) & 1) &
                                                  ~np.r_[0, ((ruled[:-1] >> 4) & 1)]))),
        "text_precision_on_verified": None if precision is None else round(precision, 4),
        "text_taps_checked_on_verified": checked,
        "text_applied": bool(apply_text), "frames_changed": int(changed.sum()),
        "seconds": round(time.time() - started, 1),
    }
    Path(f"results/label_consistency_{args.name}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report), flush=True)

    if args.apply:
        event = np.zeros(n, bool)
        event[starts[:, 0]] = True
        event[spans[:, 1]] = True
        with h5py.File(args.labels, "a") as labels:
            for key, value in (("buttons_ruled", ruled.astype(np.uint8)), ("rule_changed", changed),
                               ("rule_event", event)):
                if key in labels:
                    del labels[key]
                labels[key] = value
            labels["buttons_ruled"].attrs["text_applied"] = bool(apply_text)


if __name__ == "__main__":
    main()
