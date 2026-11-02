"""Repair the labels where replaying them stops matching the video.

The IDM's buttons are a guess; the emulator is the judge. Replay the labels
from a state that matches the video (power-on, or a save state such as the
bedroom), and at the first place they stop matching, try small edits to the
buttons -- drop a press, add an A / B / START tap, move a press or its release
by a few frames -- and keep the edit that plays in sync longest. Then go on
from there.

What comes out is buttons the emulator has checked frame by frame, up to the
first place no small edit explains: a random event the two emulators roll
differently, an edit cut, or a mistake further back than the search looks.

Search, at each loss of sync (first of RUN frames in a row under SYNC_SHARE):
  * near: every press starting in the NEAR frames before it -- removed, its
    start moved by 1-2 frames, its end by 1-64, a whole tap by 3-12 -- and A / B / START taps (and
    A / B holds of 20 or 60 frames: held A or B speeds up text) added where
    the onset head peaks or on a 6-frame grid;
  * far, only if near finds nothing: removing any tap up to FAR frames back
    (a wrong tap can sit unnoticed: the intro's false A at 1,785 shows only
    when the title screen comes up early).
Candidates are tried in order of how plausible the model finds them; the
first to play HORIZON frames past the old loss wins, else the one that got
furthest, if it beat the old loss by MARGIN.

Writes `buttons_repaired` and `verified` into the labels file, and the edits
to results/repair_<name>.json.

  uv run python -m scripts.repair_labels --frames 20000
  uv run python -m scripts.repair_labels --anchor-state states/cgb/bedroom.state --name bedroom
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

from poke.synth import BUTTONS
from poke.video import Snapper
from scripts.label_video import TAP_FRAMES
from scripts.sync_opening import (SYNC_SHARE, Replayer, align, anchored, boot, find_anchor,
                                  release_carried, video_frames)

RUN = 20
NEAR = 150
FAR = 1500
HORIZON = 600
MARGIN = 30
SNAPSHOT_EVERY = 20
TAPS = ("a", "b", "start")
GRID = 6
HOLDS = (TAP_FRAMES, 20, 60)


@dataclass
class Candidate:
    what: str
    masks: np.ndarray
    first: int            # first video frame whose buttons differ
    prior: float          # how plausible the model finds it, 0..1
    reach: int = field(default=-1)


def runs(masks: np.ndarray, k: int) -> list[tuple[int, int]]:
    """[start, stop) of every run with bit k held."""
    held = ((masks >> k) & 1).astype(bool)
    edges = np.flatnonzero(np.diff(np.r_[0, held.astype(np.int8), 0]))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def _set(masks: np.ndarray, k: int, lo: int, hi: int, on: bool) -> np.ndarray:
    out = masks.copy()
    lo, hi = max(lo, 0), min(hi, len(out))
    if on:
        out[lo:hi] |= 1 << k
    else:
        out[lo:hi] &= ~(1 << k)
    return out


def _peaks(p: np.ndarray, lo: int, hi: int, floor: float, top: int, nms: int = 3) -> list[int]:
    lo, hi = max(lo, 0), min(hi, len(p))
    out = []
    for t in np.argsort(-p[lo:hi]) + lo:
        if p[t] < floor or len(out) >= top:
            break
        if all(abs(t - u) > nms for u in out):
            out.append(int(t))
    return out


def near_candidates(masks: np.ndarray, onset_p: np.ndarray, lost: int,
                    near: int = NEAR) -> list[Candidate]:
    lo, hi = lost - near, lost + 5
    out = []
    for k, name in enumerate(BUTTONS):
        for start, stop in runs(masks, k):
            if not lo <= start < hi:
                continue
            p = float(onset_p[start, k])
            out.append(Candidate(f"drop {name} {start}-{stop}", _set(masks, k, start, stop, False),
                                 start, 1 - p))
            for d in (-2, -1, 1, 2):
                s = start + d
                if s >= stop or s < 0:
                    continue
                edited = (_set(masks, k, s, start, True) if d < 0
                          else _set(masks, k, start, s, False))
                out.append(Candidate(f"move {name} press {start}->{s}", edited, min(s, start),
                                     0.5 - 0.1 * abs(d)))
            if name in TAPS and stop - start <= 8:
                # A tap at the wrong moment: Gen 1 ignores a press while text
                # is still printing, so a few frames early is no press at all.
                for d in (-12, -9, -6, -3, 3, 6, 9, 12):
                    moved = _set(_set(masks, k, start, stop, False), k, start + d, stop + d, True)
                    out.append(Candidate(f"shift {name} {start}->{start + d}", moved,
                                         min(start, start + d), 0.45 - 0.01 * abs(d)))
            for d in (-8, -4, -2, -1, 1, 2, 4, 8, 16, 32, 64):
                e = stop + d
                if e <= start:
                    continue
                edited = (_set(masks, k, e, stop, False) if d < 0
                          else _set(masks, k, stop, e, True))
                out.append(Candidate(f"move {name} release {stop}->{e}", edited, min(e, stop),
                                     0.4 - 0.02 * abs(d)))
    for name in TAPS:
        k = BUTTONS.index(name)
        starts = set(_peaks(onset_p[:, k], lo, hi, 0.02, 12))
        if name != "start":
            starts |= set(range(max(lo, 0) - max(lo, 0) % GRID, hi, GRID))
        for t in sorted(starts):
            if not 0 <= t < len(masks) or masks[t] >> k & 1:
                continue
            # Taps, and for A / B holds: Gen 1 prints text faster while A or
            # B is held, and players hold them through long speeches.
            for length in (TAP_FRAMES,) if name == "start" else HOLDS:
                out.append(Candidate(f"add {name} {t}" + ("" if length == TAP_FRAMES else f"x{length}"),
                                     _set(masks, k, t, t + length, True), t,
                                     float(onset_p[t, k]) * (1.0 if length == TAP_FRAMES else 0.8)))
    return out


def far_candidates(masks: np.ndarray, onset_p: np.ndarray, lost: int,
                   near: int = NEAR, far: int = FAR) -> list[Candidate]:
    out = []
    for name in TAPS + ("select",):
        k = BUTTONS.index(name)
        for start, stop in runs(masks, k):
            if lost - far <= start < lost - near:
                out.append(Candidate(f"drop {name} {start}-{stop}",
                                     _set(masks, k, start, stop, False), start,
                                     1 - float(onset_p[start, k])))
    return out


def latest(snapshots: list[tuple[int, tuple]], frame: int):
    """The last snapshot taken before video frame `frame` was pressed."""
    usable = [s for s in snapshots if s[0] <= frame]
    return usable[-1] if usable else snapshots[0]


class Search:
    def __init__(self, replayer: Replayer, start: tuple, limit: int, run: int = RUN,
                 judge_from: int = 0):
        self.rep, self.limit, self.run = replayer, limit, run
        # Frames before this are played but not judged: from power-on, the
        # boot screens of PyBoy and the capture emulator differ for ~50
        # frames whatever is pressed.
        self.judge_from = judge_from
        self.snapshots = [(self._next(), start)]
        self.played = 0

    def _next(self) -> int:
        return self.rep.e + self.rep.offset

    def play(self, snapshot: tuple, masks: np.ndarray, stop_at: int, record: bool = False) -> int:
        """Video frame where sync is lost (first of `run` frames under
        SYNC_SHARE), or `stop_at`."""
        self.rep.load(snapshot)
        self.rep.masks = masks
        bad, first_bad = 0, None
        while True:
            if record and self._next() % SNAPSHOT_EVERY == 0 and self._next() > self.snapshots[-1][0]:
                self.snapshots.append((self._next(), self.rep.save()))
            out = self.rep.step()
            self.played += 1
            if out is None:
                if self._next() >= stop_at:
                    return stop_at
                continue
            v, share = out
            if v >= stop_at:
                return stop_at
            if share < SYNC_SHARE and v >= self.judge_from:
                bad += 1
                first_bad = v if first_bad is None else first_bad
                if bad >= self.run:
                    return first_bad
            else:
                bad, first_bad = 0, None

    def base(self, masks: np.ndarray, frame: int) -> int:
        """Play the current labels from the last snapshot before `frame`,
        keeping snapshots on the way; drop the ones after it first."""
        self.snapshots = [s for s in self.snapshots if s[0] <= frame] or self.snapshots[:1]
        return self.play(self.snapshots[-1][1], masks, self.limit, record=True)

    def best(self, candidates: list[Candidate], lost: int, exclude: frozenset = frozenset(),
             avoid_reach: int | None = None) -> Candidate | None:
        """The edit that plays furthest; `avoid_reach` skips edits that lose
        sync exactly where an undone edit did (in effect the same edit: a B
        tap one frame later)."""
        cap = min(self.limit, lost + HORIZON)
        winner = None
        for c in sorted(candidates, key=lambda c: -c.prior):
            if c.what in exclude:
                continue
            c.reach = self.play(latest(self.snapshots, c.first)[1], c.masks, cap)
            if c.reach == avoid_reach:
                continue
            if winner is None or c.reach > winner.reach:
                winner = c
            if c.reach >= cap:
                break
        return winner if winner and winner.reach >= lost + MARGIN else None


def repair(search: Search, masks: np.ndarray, onset_p: np.ndarray, lost: int, limit: int,
           max_edits: int, max_backtracks: int):
    """Greedy repair with backtracking: (masks, video frame sync is lost at, edits).

    Each step takes the best edit at the current loss of sync. When no edit
    helps, the most recent edits are undone one at a time and the next-best
    edit at that point is tried instead -- a weak early edit (a tap that buys
    40 frames) can put the emulator on a wrong track that only shows later.
    The furthest-reaching labels found are returned.
    """
    def find(m, at, exclude, avoid=None):
        found = search.best(near_candidates(m, onset_p, at), at, exclude, avoid)
        return found or search.best(far_candidates(m, onset_p, at), at, exclude, avoid)

    # stack entries: (masks before the edit, sync lost before it, edit, tried edits)
    stack: list[tuple[np.ndarray, int, Candidate, set]] = []
    best = (masks, lost, [])
    backtracks = 0
    while lost < limit and len(stack) < max_edits:
        t = time.time()
        winner = find(masks, lost, frozenset())
        tried: set = set()
        if winner is None:
            # Undo edits until one has an untried alternative that helps.
            undone_first = lost
            while stack and winner is None and backtracks < max_backtracks:
                masks, lost, undone, tried = stack.pop()
                undone_first = min(undone_first, undone.first)
                tried.add(undone.what)
                backtracks += 1
                search.base(masks, undone_first)
                winner = find(masks, lost, frozenset(tried), undone.reach)
                print(f"  backtrack: undo {undone.what} -> "
                      f"{winner.what if winner else 'no alternative'}", flush=True)
            if winner is None:
                print(f"stuck at video frame {best[1]}: no edit plays {MARGIN} frames longer",
                      flush=True)
                break
            first = min(winner.first, undone_first)
        else:
            first = winner.first
        stack.append((masks, lost, winner, tried))
        masks = winner.masks
        new_lost = search.base(masks, first)
        print(f"{winner.what:32s} sync {lost} -> {new_lost}  ({time.time() - t:.0f}s, "
              f"{search.played} frames played)", flush=True)
        lost = new_lost
        if lost > best[1]:
            best = (masks, lost, [{"edit": e.what, "at": e.first, "lost_before": lb,
                                   "prior": round(e.prior, 3)} for _, lb, e, _ in stack])
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", default="data/longplay/video.h5")
    parser.add_argument("--labels", default="data/longplay/labels.h5")
    parser.add_argument("--frames", type=int, default=20_000, help="video frames to repair")
    parser.add_argument("--anchor-state", default=None)
    parser.add_argument("--anchor-search", type=int, nargs=2, default=(6000, 7600))
    parser.add_argument("--buttons", default="auto",
                        help="labels to start from: auto = buttons_ruled when present, else buttons")
    parser.add_argument("--max-edits", type=int, default=200)
    parser.add_argument("--max-backtracks", type=int, default=12,
                        help="times to undo recent edits and try the next-best one when stuck")
    parser.add_argument("--settle", type=int, default=150,
                        help="video frames after the start that are not judged (boot screens)")
    parser.add_argument("--name", default="poweron")
    parser.add_argument("--dry-run", action="store_true", help="do not write the labels file")
    args = parser.parse_args()

    started = time.time()
    with h5py.File(args.video) as h5, h5py.File(args.labels) as labels:
        snapper = Snapper(h5["colours"][:])
        key = args.buttons if args.buttons != "auto" else (
            "buttons_ruled" if "buttons_ruled" in labels else "buttons")
        masks = labels[key][:].astype(np.int64)
        print(f"starting from {key}", flush=True)
        onset_p = labels["onset_p"][:].astype(np.float32)
        env = None
        if args.anchor_state:
            env = anchored(args.anchor_state)
            pyboy = env.pyboy
            anchor, share = find_anchor(env, h5, snapper, *args.anchor_search)
            offset = anchor + 1
            masks = release_carried(masks, offset)
            first_frame = offset
            print(f"anchor {args.anchor_state} = video frame {anchor} ({share:.4f})", flush=True)
        else:
            offset, _ = align(h5, snapper)
            pyboy = boot()
            first_frame = 0
        limit = first_frame + args.frames
        video = video_frames(h5, 0, limit + 8)
    original = masks.copy()
    replayer = Replayer(pyboy, snapper, video, masks, offset)
    search = Search(replayer, replayer.save(), limit,
                    judge_from=first_frame + (0 if args.anchor_state else args.settle))
    lost = search.base(masks, first_frame)
    print(f"labels as decoded: in sync until video frame {lost}", flush=True)
    masks, lost, edits = repair(search, masks, onset_p, lost, limit, args.max_edits,
                                args.max_backtracks)
    if env:
        env.close()
    else:
        pyboy.stop(save=False)

    changed = int((masks != original).sum())
    report = {"name": args.name, "start": first_frame, "verified_until": int(lost),
              "limit": limit, "edits": edits, "frames_changed": changed,
              "anchor_state": args.anchor_state, "seconds": round(time.time() - started)}
    print(f"verified video frames {first_frame}-{lost} with {len(edits)} edits "
          f"({changed} frames changed) in {report['seconds']}s", flush=True)
    Path(f"results/repair_{args.name}.json").write_text(json.dumps(report, indent=1))
    if not args.dry_run:
        with h5py.File(args.labels, "a") as labels:
            if "buttons_repaired" not in labels:
                labels["buttons_repaired"] = labels[key][:]
                labels["verified"] = np.zeros(len(labels["buttons"]), bool)
            span = slice(first_frame, lost)
            labels["buttons_repaired"][span] = masks[span].astype(np.uint8)
            verified = labels["verified"][:]
            verified[span] = True
            labels["verified"][:] = verified
            labels["verified"].attrs[f"repair_{args.name}"] = json.dumps(
                {"start": first_frame, "until": int(lost), "edits": len(edits)})


if __name__ == "__main__":
    main()
