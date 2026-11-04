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
import os
from multiprocessing import get_context

import numpy as np
from pyboy import PyBoy

from poke.env import SRAM_BYTES
from poke.video import Snapper
from scripts.sync_opening import ROM, TEXT_LIMIT, TEXT_SLACK, Replayer

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
