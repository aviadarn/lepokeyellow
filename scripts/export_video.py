"""Export the labelled longplay as training data, in two formats.

Inputs: data/longplay/video.h5 (frames) and data/longplay/labels.h5 (the
IDM's per-frame buttons, poke.synth.BUTTONS order). The buttons exported are
the best each frame has: `buttons_repaired` where scripts/repair_labels.py
verified them against the emulator (`verified` True, confidence 1), else
`buttons_ruled` (scripts/check_labels.py: every answered ▼ has a press),
else the decoded ones. Each step also counts `rule_events` (steps and ▼
answers the screen shows inside it) and `rule_fixed` (frames the rules
changed).

Segments: the video is split wherever the cut head fires (cut_p > 0.5) or a
frame is unreadable (labels state 255); no step spans a split.

1. PokeDreamer schema (data/longplay/pokedreamer/transitions_XXXX.npz),
   the same arrays as poke.dreamer.SCHEMA's PokeDreamer block:
     * one step = 24 video frames (= game frames), obs = the step's first frame;
     * action = the step's press: the earliest A / B / START onset, else the
       direction held longest if held >= 8 frames, else pass. SELECT is not
       in PokeDreamer's action set and becomes pass (counted);
     * RAM probes cannot be read off a video: map_ids / xs / ys / facings /
       badges are 255 and the HP arrays -1, with `probes_known` False.
       rewards are 0;
     * extra: `frame_index` (video frame of obs), `segment_ids`,
       `held_masks` (N, 24) u1 -- the per-frame labels the action was
       chosen from, nothing thrown away -- `action_confidence`, and
       `verified` (every frame of the step replay-verified).
2. lepokeyellow SCHEMA 3 (data/longplay/longplay_schema3.h5) through
   poke.writer.EpisodeWriter: 60-frame steps, grey frames, one action of
   noop/up/down/left/right/a/b chosen the same way (START/SELECT -> noop).
   The env's step is hold 20 + settle 40; a human hold need not fit that, so
   `action_confidence` is written beside it and the attrs say what this is.

  uv run python -m scripts.export_video
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from poke.dreamer import CHUNK_STEPS, PASS
from poke.synth import BUTTONS

STEP = 24
OBS_BATCH = 512
LONG_HOLD = 8
#: poke.synth.BUTTONS index -> PokeDreamer action index.
TO_DREAMER = {"down": 0, "left": 1, "right": 2, "up": 3, "a": 4, "b": 5, "start": 6}
DIRECTIONS = ("up", "down", "left", "right")
TAPS = ("a", "b", "start")
#: DMG greys the repo's frames use, and the CGB-colour -> grey rule.
DMG_GREYS = np.array([0, 85, 153, 255])


def segments(cut_p: np.ndarray, state: np.ndarray, min_frames: int) -> list[tuple[int, int]]:
    """[start, stop) runs with no cut and no unreadable frame."""
    broken = (cut_p > 0.5) | (state == 255).all(axis=1)
    edges = np.flatnonzero(broken)
    bounds, start = [], 0
    for edge in list(edges) + [len(cut_p)]:
        if edge - start >= min_frames:
            bounds.append((start, int(edge)))
        start = int(edge) + 1 if edge < len(cut_p) and (state[edge] == 255).all() else int(edge)
    return bounds


def choose(held: np.ndarray, held_p: np.ndarray, before: np.ndarray, names=BUTTONS):
    """One action for a block of frames: (button name or None, confidence).

    held: (T, 8) bool; before: (8,) bool, the frame before the block (so a
    press that started earlier is not an onset here).
    """
    previous = np.vstack([before[None], held[:-1]])
    onsets = held & ~previous
    first = {}
    for k, name in enumerate(names):
        hits = np.flatnonzero(onsets[:, k])
        if hits.size:
            first[name] = int(hits[0])
    taps = [name for name in TAPS if name in first]
    if taps:
        name = min(taps, key=lambda n: first[n])
        k = names.index(name)
        return name, float(held_p[first[name], k])
    counts = {name: int(held[:, names.index(name)].sum()) for name in DIRECTIONS}
    name = max(counts, key=counts.get)
    if counts[name] >= LONG_HOLD:
        k = names.index(name)
        return name, float(held_p[held[:, k], k].mean())
    # Nothing pressed: confidence that everything really was released.
    return None, float(1 - held_p.max(axis=1).mean())


def greys(colours: np.ndarray) -> np.ndarray:
    """Per-colour DMG grey, by luminance. Approximate: Yellow's CGB palettes
    do not always order the way the DMG shades of the same tiles do."""
    luminance = colours.astype(np.float32) @ np.array([0.299, 0.587, 0.114])
    return DMG_GREYS[np.abs(luminance[:, None] - DMG_GREYS[None]).argmin(1)].astype(np.uint8)


def frames_at(video, refs, colours, index: np.ndarray) -> np.ndarray:
    rows = refs[index]
    low = int(rows.min())
    block = video["uframes"][low:int(rows.max()) + 1]
    return colours[block[rows - low]]


def label_arrays(labels):
    """(held (N, 8) bool, confidence source (N, 8) f4, verified (N,) bool,
    rule_changed (N,) bool, rule_event (N,) bool).

    Buttons come from the best source each frame has: the emulator-repaired
    labels where `verified`, else the rule-checked ones (`buttons_ruled`,
    scripts/check_labels.py), else the decoded ones; labels files from before
    decoding fall back to held_p > 0.5. Verified frames are certain: their
    held_p is replaced by the held bits.
    """
    held_p = labels["held_p"][:].astype(np.float32)
    n = len(held_p)
    verified = labels["verified"][:] if "verified" in labels else np.zeros(n, bool)
    key = "buttons_ruled" if "buttons_ruled" in labels else "buttons"
    if key in labels:
        masks = labels[key][:].astype(np.int64)
        if "buttons_repaired" in labels:
            masks[verified] = labels["buttons_repaired"][:][verified]
        held = ((masks[:, None] >> np.arange(8)) & 1).astype(bool)
    else:
        held = held_p > 0.5
    held_p[verified] = held[verified]
    changed = labels["rule_changed"][:] if "rule_changed" in labels else np.zeros(n, bool)
    events = labels["rule_event"][:] if "rule_event" in labels else np.zeros(n, bool)
    return held, held_p, verified, changed, events


def export_dreamer(video, labels, bounds, out: Path) -> dict:
    refs = video["frame_ref"][:]
    colours = video["colours"][:]
    held, held_p, verified, changed, events = label_arrays(labels)
    out.mkdir(parents=True, exist_ok=True)
    rows: dict[str, list] = {k: [] for k in ("obs", "actions", "episode_starts", "frame_index",
                                               "segment_ids", "held_masks", "action_confidence",
                                               "verified", "rule_events", "rule_fixed")}
    written, select_dropped, steps = [], 0, 0

    def flush(final=False):
        nonlocal rows
        n = len(rows["actions"])
        while n >= CHUNK_STEPS or (final and n):
            take = min(n, CHUNK_STEPS)
            arrays = {
                "obs": np.stack(rows["obs"][:take]),
                "actions": np.array(rows["actions"][:take], np.uint8),
                "rewards": np.zeros(take, np.float32),
                "episode_starts": np.array(rows["episode_starts"][:take], bool),
                "map_ids": np.full(take, 255, np.uint8), "xs": np.full(take, 255, np.uint8),
                "ys": np.full(take, 255, np.uint8), "facings": np.full(take, 255, np.uint8),
                "in_battles": np.zeros(take, bool), "dialog_opens": np.zeros(take, bool),
                "badges": np.full(take, 255, np.uint8),
                "party_hps": np.full((take, 6), -1, np.int16),
                "party_max_hps": np.full((take, 6), -1, np.int16),
                "probes_known": np.zeros(take, bool),
                "frame_index": np.array(rows["frame_index"][:take], np.int32),
                "segment_ids": np.array(rows["segment_ids"][:take], np.int32),
                "held_masks": np.stack(rows["held_masks"][:take]).astype(np.uint8),
                "action_confidence": np.array(rows["action_confidence"][:take], np.float16),
                "verified": np.array(rows["verified"][:take], bool),
                "rule_events": np.array(rows["rule_events"][:take], np.uint8),
                "rule_fixed": np.array(rows["rule_fixed"][:take], np.uint8),
            }
            path = out / f"transitions_{len(written) + 1:04d}.npz"
            np.savez_compressed(path, **arrays)
            written.append(str(path))
            rows = {k: v[take:] for k, v in rows.items()}
            n -= take

    bits = 1 << np.arange(8)
    for segment, (start, stop) in enumerate(bounds):
        starts = np.arange(start, stop - STEP + 1, STEP)
        if not len(starts):
            continue
        for k, s in enumerate(starts):
            # Read OBS_BATCH steps' frames at a time and flush full chunks as
            # they fill: one read per segment spans every unique frame in it
            # (all 12 GB of uframes if the video has few cuts).
            if k % OBS_BATCH == 0:
                obs = frames_at(video, refs, colours, starts[k:k + OBS_BATCH])
                flush()
            block = held[s:s + STEP]
            before = held[s - 1] if s > start else np.zeros(8, bool)
            name, confidence = choose(block, held_p[s:s + STEP], before)
            if name is None and block[:, BUTTONS.index("select")].any():
                select_dropped += 1
            rows["obs"].append(obs[k % OBS_BATCH])
            rows["actions"].append(PASS if name is None else TO_DREAMER[name])
            rows["episode_starts"].append(k == 0)
            rows["frame_index"].append(int(s))
            rows["segment_ids"].append(segment)
            rows["held_masks"].append((block * bits).sum(1))
            rows["action_confidence"].append(confidence)
            rows["verified"].append(bool(verified[s:s + STEP].all()))
            rows["rule_events"].append(int(events[s:s + STEP].sum()))
            rows["rule_fixed"].append(int(changed[s:s + STEP].sum()))
            steps += 1
        flush()
    flush(final=True)
    return {"steps": steps, "chunks": len(written), "select_only_steps_as_pass": select_dropped,
            "verified_steps": int(sum(int(np.load(w)["verified"].sum()) for w in written))}


def export_schema3(video, labels, bounds, path: Path) -> dict:
    from poke.actions import ACTIONS, to_multihot
    from poke.writer import EpisodeWriter

    refs = video["frame_ref"][:]
    grey = greys(video["colours"][:])
    held, held_p, verified, changed, events = label_arrays(labels)
    confidence, checked, rule_events, rule_fixed = [], [], [], []
    with EpisodeWriter(path, rom="pokeyellow.gbc", state="pokeyellow.gbc", act_frames=60,
                       seed=0, game="yellow", task="longplay_video") as writer:
        for start, stop in bounds:
            starts = np.arange(start, stop - 60 + 1, 60)
            if len(starts) < 2:
                continue
            for s in starts:
                before = held[s - 1] if s > start else np.zeros(8, bool)
                name, conf = choose(held[s:s + 60], held_p[s:s + 60], before)
                action = ACTIONS.index(name) if name in ACTIONS else ACTIONS.index("noop")
                rows = refs[[s]]
                frame = grey[video["uframes"][int(rows[0])]][..., None]
                writer.add_step(frame, to_multihot(action), party_count=255)
                confidence.append(conf)
                checked.append(bool(verified[s:s + 60].all()))
                rule_events.append(int(events[s:s + 60].sum()))
                rule_fixed.append(int(changed[s:s + 60].sum()))
            writer.end_episode(kind="video", name=f"segment_{start}")
    with h5py.File(path, "a") as h5:
        h5["action_confidence"] = np.array(confidence, np.float16)
        h5["verified"] = np.array(checked, bool)
        h5["rule_events"] = np.array(rule_events, np.uint8)
        h5["rule_fixed"] = np.array(rule_fixed, np.uint8)
        h5.attrs["source"] = "video:aqr_YgqF2vM, IDM labels"
        h5.attrs["note"] = ("60-frame steps cut from human play, not env steps; party_counts "
                            "255 = unknown; location -1; state_sha1 is a placeholder")
    return {"steps": len(confidence), "verified_steps": int(sum(checked))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", default="data/longplay/video.h5")
    parser.add_argument("--labels", default="data/longplay/labels.h5")
    parser.add_argument("--out", default="data/longplay")
    parser.add_argument("--min-segment", type=int, default=240, help="frames")
    args = parser.parse_args()

    out = Path(args.out)
    with h5py.File(args.video) as video, h5py.File(args.labels) as labels:
        n = len(labels["held_p"])
        # Confirmed cuts when the labels have them (label_video.confirm_cuts).
        cut = labels["cut"][:] if "cut" in labels else labels["cut_p"][:]
        bounds = segments(cut.astype(np.float32), labels["state"][:],
                          args.min_segment)
        print(f"{len(bounds)} segments covering {sum(b - a for a, b in bounds)} of {n} frames",
              flush=True)
        report = {"segments": len(bounds),
                  "frames_in_segments": int(sum(b - a for a, b in bounds)),
                  "frames": int(n)}
        report["pokedreamer"] = export_dreamer(video, labels, bounds, out / "pokedreamer")
        print("pokedreamer:", report["pokedreamer"], flush=True)
        report["schema3"] = export_schema3(video, labels, bounds, out / "longplay_schema3.h5")
        print("schema3:", report["schema3"], flush=True)
    (out / "export.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
