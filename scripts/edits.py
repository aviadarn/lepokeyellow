"""Edits to labelled buttons: one grammar for every button and every kind
of miss, ranked by how plausible the IDM finds each.

A label is a run of frames with one button held -- a press. An edit adds a
press, removes one, moves its start or its end (by up to 256 frames either
way, so a hold the model read as 157 frames can become the tap it was),
sets its length, splits it in two, or turns it into another button. Nothing
here knows about walking, menus or text: the emulator judges every edit
(scripts/repair_labels.py), and the IDM only decides which to try first.

A Candidate keeps its edit as a few interval writes on a shared base, so a
search over thousands of them never holds thousands of label arrays.
"""

from __future__ import annotations

import numpy as np

from poke.synth import BUTTONS
from scripts.label_video import TAP_FRAMES

OFFSETS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
LENGTHS = (TAP_FRAMES, 20, 60)
GRID = 6
PEAKS = 6
SPLITS = 3
FLOOR = 0.01


class Candidate:
    """An edit: `what` it is, the first frame it changes, the IDM's prior,
    and how far it plays in sync once judged (`reach`). Its masks are the
    base with `ops` -- (button, lo, hi, on) writes -- applied, built on
    demand; or given outright (`masks=`)."""

    def __init__(self, what: str, masks: np.ndarray | None = None, first: int = 0,
                 prior: float = 0.0, reach: int = -1, *, base: np.ndarray | None = None,
                 ops: tuple = ()):
        self.what, self.first, self.prior, self.reach = what, first, prior, reach
        self._masks, self.base, self.ops = masks, base, ops

    @property
    def masks(self) -> np.ndarray:
        if self._masks is not None:
            return self._masks
        out = self.base.copy()
        for k, lo, hi, on in self.ops:
            out = _write(out, k, lo, hi, on)
        return out

    def __repr__(self) -> str:
        return f"Candidate({self.what!r}, first={self.first}, prior={self.prior:.3f}, reach={self.reach})"


def runs(masks: np.ndarray, k: int) -> list[tuple[int, int]]:
    """[start, stop) of every run with bit k held."""
    held = ((masks >> k) & 1).astype(bool)
    edges = np.flatnonzero(np.diff(np.r_[0, held.astype(np.int8), 0]))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def _write(out: np.ndarray, k: int, lo: int, hi: int, on: bool) -> np.ndarray:
    lo, hi = max(lo, 0), min(hi, len(out))
    if on:
        out[lo:hi] |= 1 << k
    else:
        out[lo:hi] &= ~(1 << k)
    return out


def _set(masks: np.ndarray, k: int, lo: int, hi: int, on: bool) -> np.ndarray:
    return _write(masks.copy(), k, lo, hi, on)


def _peaks(p: np.ndarray, lo: int, hi: int, floor: float, top: int, nms: int = 3) -> list[int]:
    lo, hi = max(lo, 0), min(hi, len(p))
    out = []
    for t in np.argsort(-p[lo:hi]) + lo:
        if p[t] < floor or len(out) >= top:
            break
        if all(abs(t - u) > nms for u in out):
            out.append(int(t))
    return out


def _log_q(held_p: np.ndarray, lo: int, hi: int, k: int, on: bool) -> np.ndarray:
    p = np.clip(held_p[lo:hi, k].astype(np.float64), FLOOR, 1 - FLOOR)
    return np.log(p if on else 1 - p)


def prior(masks: np.ndarray, edited: np.ndarray, held_p: np.ndarray, lo: int, hi: int) -> float:
    """Log-likelihood gain of `edited` over `masks` on frames [lo, hi) under
    the IDM's per-frame held probabilities (clipped to 0.01..0.99)."""
    total = 0.0
    for k in range(8):
        old = (masks[lo:hi] >> k & 1).astype(bool)
        new = (edited[lo:hi] >> k & 1).astype(bool)
        changed = np.flatnonzero(old != new)
        if changed.size:
            p = np.clip(held_p[lo + changed, k].astype(np.float64), FLOOR, 1 - FLOOR)
            total += float(np.sum(np.where(new[changed], np.log(p), np.log(1 - p))
                                  - np.where(old[changed], np.log(p), np.log(1 - p))))
    return total


def _op_prior(masks: np.ndarray, held_p: np.ndarray, ops: tuple) -> float:
    """`prior` of applying `ops`, computed on the frames they touch only.
    The ops of one edit never touch the same (frame, button) twice."""
    total = 0.0
    for k, lo, hi, on in ops:
        lo, hi = max(lo, 0), min(hi, len(masks))
        if hi <= lo:
            continue
        was = (masks[lo:hi] >> k & 1).astype(bool)
        flips = was != on
        if flips.any():
            total += float(np.sum((_log_q(held_p, lo, hi, k, on) - _log_q(held_p, lo, hi, k, not on))[flips]))
    return total


def candidates(masks: np.ndarray, held_p: np.ndarray, onset_p: np.ndarray, lost: int,
               window: int) -> list[Candidate]:
    """Every edit touching frames [lost - window, lost + 5)."""
    n = len(masks)
    lo, hi = max(lost - window, 0), min(lost + 5, n)
    out: list[Candidate] = []

    def add(what: str, first: int, ops: tuple) -> None:
        out.append(Candidate(what, first=max(first, 0), prior=_op_prior(masks, held_p, ops),
                             base=masks, ops=ops))

    for k, name in enumerate(BUTTONS):
        starts = set(range(lo + (-lo) % GRID, hi, GRID)) | set(_peaks(onset_p[:, k], lo, hi, 0.0, PEAKS))
        for t in sorted(starts):
            if masks[t] >> k & 1:
                continue
            for length in LENGTHS:
                what = f"add {name} {t}" + ("" if length == TAP_FRAMES else f"x{length}")
                add(what, t, ((k, t, t + length, True),))
        for s, e in runs(masks, k):
            if e <= lo or s >= hi:
                continue
            add(f"drop {name} {s}-{e}", s, ((k, s, e, False),))
            for d in OFFSETS:
                for s2 in (s - d, s + d):
                    if 0 <= s2 < e and s2 != s:
                        ops = ((k, s2, s, True),) if s2 < s else ((k, s, s2, False),)
                        add(f"start {name} {s}->{s2}", min(s, s2), ops)
                for e2 in (e - d, e + d):
                    if s < e2 <= n and e2 != e:
                        ops = ((k, e2, e, False),) if e2 < e else ((k, e, e2, True),)
                        add(f"end {name} {e}->{e2}", min(e, e2), ops)
            for length in LENGTHS:
                e2 = min(s + length, n)
                if e2 != e:
                    ops = ((k, e2, e, False),) if e2 < e else ((k, e, e2, True),)
                    add(f"length {name} {s}-{e}->{length}", min(e, e2), ops)
            if e - s >= 6:
                cuts = set(_peaks(onset_p[:, k], s + 3, e - 2, 0.0, SPLITS)) | {(s + e) // 2}
                for t in sorted(cuts):
                    add(f"split {name} {s}-{e}@{t}", t - 2, ((k, t - 2, t, False),))
            for j, other in enumerate(BUTTONS):
                if j != k:
                    add(f"swap {name}->{other} {s}-{e}", s, ((k, s, e, False), (j, s, e, True)))
    return out


def _group(c: Candidate) -> tuple[str, str]:
    kind, rest = c.what.split(" ", 1)
    return kind, rest.split(" ")[0].split("->")[0]


BANDS = (30, 100, 300)    # frames before the loss: edits nearer it are likelier causes


def _band(c: Candidate, lost: int | None) -> int:
    if lost is None:
        return 0
    distance = max(lost - c.first, 0)
    return next((i for i, edge in enumerate(BANDS) if distance <= edge), len(BANDS))


def screen(cands: list[Candidate], cap: int, lost: int | None = None) -> list[Candidate]:
    """At most about `cap` candidates. First the top few of every (kind,
    button) group and, given `lost`, every distance band before it -- the
    model can be confidently wrong, so no kind of fix may be crowded out,
    and a flat prior must not push out the edits next to the loss. Then the
    rest. Order: likelier first, then nearer the loss."""
    def order(c):
        return (-c.prior, abs(lost - c.first) if lost is not None else 0)

    ranked = sorted(cands, key=order)
    per_group = max(1, cap // (48 * (len(BANDS) + 1 if lost is not None else 1)))
    per_group = max(per_group, 2 if lost is None else 1)
    kept, counts = [], {}
    for c in ranked:
        g = _group(c) + (_band(c, lost),)
        if counts.get(g, 0) < per_group:
            counts[g] = counts.get(g, 0) + 1
            kept.append(c)
    chosen = {id(c) for c in kept}
    for c in ranked:
        if len(kept) >= cap:
            break
        if id(c) not in chosen:
            kept.append(c)
    return sorted(kept, key=order)
