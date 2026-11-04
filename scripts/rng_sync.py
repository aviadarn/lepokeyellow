"""Match the game's random numbers to the video's.

Gen 1 keeps its random state in two bytes of HRAM, `hRandomAdd` and
`hRandomSub`; the VBlank handler stirs them every frame with the hardware
timer. NPC steps, wild encounters and battle rolls all draw on them, so once
the two emulators disagree on them nothing a button can do makes their
screens agree again. `find_rng` writes candidate values at a snapshot and
keeps those that replay the next frames as the video shows them.

A sprite in the wrong place is about 1 % of the screen, inside what the
pixel share forgives (`SYNC_SHARE`), so the search holds every frame to a
stricter share as well as the text check.
"""

from __future__ import annotations

import io
import itertools
import os
import time
from multiprocessing import get_context

import numpy as np
from pyboy import PyBoy

from poke.env import SRAM_BYTES
from poke.video import Snapper
from scripts.sync_opening import ROM, TEXT_LIMIT, TEXT_SLACK, Replayer, in_sync

RANDOM_ADD = 0xFFD3
RANDOM_SUB = 0xFFD4
STRICT_SHARE = 0.995


def set_rng(pyboy, add: int, sub: int) -> None:
    pyboy.memory[RANDOM_ADD] = add
    pyboy.memory[RANDOM_SUB] = sub


def _matching(job) -> list[tuple[int, int]]:
    blob, tracker, video, masks, colours, slack, patience, horizon, share, values = job
    os.nice(19)
    pyboy = PyBoy(ROM, window="null", ram_file=io.BytesIO(bytes(SRAM_BYTES)))
    replayer = Replayer(pyboy, Snapper(colours), video, masks, tracker[1], slack, patience)
    out = []
    for add, sub in values:
        replayer.load((blob, tracker))
        set_rng(pyboy, add, sub)
        for _ in range(horizon):
            step = replayer.step()
            if step is None:
                continue
            _, frame_share, text = step
            if frame_share < share or text is None or text >= TEXT_LIMIT:
                break
        else:
            out.append((int(add), int(sub)))
    pyboy.stop(save=False)
    return out


def find_rng(replayer: Replayer, snapshot: tuple, horizon: int, values=None,
             workers: int = 4, share: float = STRICT_SHARE) -> list[tuple[int, int]]:
    """Every (add, sub) that, written at `snapshot`, keeps all of the next
    `horizon` frames matching the video: share >= `share` and the text check
    passed. `values` defaults to all 65,536; they are split over `workers`
    processes at nice 19, each with its own emulator."""
    if values is None:
        values = [(a, s) for a in range(256) for s in range(256)]
    blob, (e, offset, streak, pending, held) = snapshot
    # Only the video and labels the horizon reads, re-indexed from `first`.
    first = max(e + offset - replayer.slack - TEXT_SLACK, 0)
    last = e + offset + horizon + replayer.slack + TEXT_SLACK + 1
    video, masks = replayer.video[first:last], replayer.masks[first:last]
    tracker = (e, offset - first, streak, pending, held)
    chunks = [values[i::workers] for i in range(workers)]
    jobs = [(blob, tracker, video, masks, replayer.snapper.colours, replayer.slack,
             replayer.patience, horizon, share, chunk) for chunk in chunks if chunk]
    with get_context("spawn").Pool(len(jobs)) as pool:
        found = [v for part in pool.map(_matching, jobs) for v in part]
    return sorted(found)


def _reaches(job) -> list[tuple[int, tuple]]:
    (blob, tracker, video, masks, colours, slack, patience, writes, at, addresses, combos,
     stop, judge_from, run, first) = job
    os.nice(19)
    pyboy = PyBoy(ROM, window="null", ram_file=io.BytesIO(bytes(SRAM_BYTES)))
    replayer = Replayer(pyboy, Snapper(colours), video, masks, tracker[1], slack, patience)
    out = []
    for combo in combos:
        replayer.load((blob, tracker))
        replayer.writes = dict(writes)
        replayer.writes[at] = replayer.writes.get(at, ()) + tuple(zip(addresses, combo))
        bad, first_bad, reach = 0, None, stop
        while True:
            step = replayer.step()
            if step is None:
                if replayer.e + replayer.offset >= stop:
                    break
                continue
            v, share, text = step
            if v >= stop:
                break
            if not in_sync(share, text) and v >= judge_from:
                bad += 1
                first_bad = v if first_bad is None else first_bad
                if bad >= run:
                    reach = first_bad
                    break
            else:
                bad, first_bad = 0, None
        out.append((reach + first, tuple(int(x) for x in combo)))
    pyboy.stop(save=False)
    return out


def find_state(replayer: Replayer, snapshot: tuple, cap: int, registers, workers: int = 4,
               writes: dict | None = None, at: int | None = None, judge_from: int = 0,
               run: int = 20, deadline: float | None = None) -> list[tuple[int, tuple]]:
    """Every combination of `registers` ((address, values), ...) written at
    video frame `at` (default: the snapshot's), ranked by how far it plays in
    sync (first of `run` frames out of sync, or `cap`): [(reach, values)],
    furthest first. `writes` are the state writes already made. Split over
    `workers` processes at nice 19, each with its own emulator, in small
    chunks; past `deadline` (wall clock) no more chunks are taken, and the
    values tried so far are ranked."""
    blob, (e, offset, streak, pending, held) = snapshot
    at = e + offset if at is None else at
    first = max(e + offset - replayer.slack - TEXT_SLACK, 0)
    last = cap + replayer.slack + TEXT_SLACK + 1
    video, masks = replayer.video[first:last], replayer.masks[first:last]
    tracker = (e, offset - first, streak, pending, held)
    shifted = {f - first: pairs for f, pairs in (writes or {}).items() if first <= f < last}
    addresses = tuple(a for a, _ in registers)
    combos = list(itertools.product(*[list(values) for _, values in registers]))
    size = max(1, min(256, -(-len(combos) // workers)))
    chunks = [combos[i:i + size] for i in range(0, len(combos), size)]
    jobs = [(blob, tracker, video, masks, replayer.snapper.colours, replayer.slack,
             replayer.patience, shifted, at - first, addresses, chunk, cap - first,
             judge_from - first, run, first) for chunk in chunks]
    ranked: list[tuple[int, tuple]] = []
    if deadline is not None and time.time() > deadline:
        return ranked
    with get_context("spawn").Pool(min(workers, len(jobs))) as pool:
        for part in pool.imap_unordered(_reaches, jobs):
            ranked.extend(part)
            if deadline is not None and time.time() > deadline:
                pool.terminate()
                break
    return sorted(ranked, key=lambda r: (-r[0], r[1]))
