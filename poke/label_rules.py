"""What the screen proves about the buttons, anywhere in a video.

The IDM's labels are a guess, and replaying them in the emulator only checks
the opening of the longplay (past it, edit cuts and the random number
generator part the two games). Two things the picture itself shows hold
everywhere, and each says a button was pressed:

  * walking. The overworld scrolls 2 px a frame while the player walks, 8
    frames a step, in the walking direction; a step that starts from
    standing has its direction held just before it (true labels: 98.7 % of
    155 steps, within frames t-3 .. t-1 of the first scrolling frame t);
  * text. A `▼` prompt waits for A; B does not answer it.

Shifts are `(dy, dx)` with new[y, x] == old[y + dy, x + dx]; directions are
button bits in poke.synth.BUTTONS order (up, down, left, right, ...).
"""

from __future__ import annotations

import numpy as np

#: Scroll shift between consecutive frames -> the direction button walked.
SHIFTS: dict[tuple[int, int], int] = {(-2, 0): 0, (2, 0): 1, (0, -2): 2, (0, 2): 3}


def _overlap(frames: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """(n-1,) share of the overlap where frame t equals frame t-1 shifted."""
    h, w = frames.shape[1:3]
    old = frames[:-1, max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)]
    new = frames[1:, max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)]
    return (old == new).mean(axis=(1, 2))


def scroll(frames: np.ndarray, match: float = 0.97) -> np.ndarray:
    """(n,) int8: the direction bit frame t scrolled by against frame t-1, or -1.

    A frame scrolls only when the shifted overlap matches on at least
    `match` of its pixels and the frames do not already match unshifted --
    a sprite walking, a menu opening or a battle slide-in moves part of the
    screen, not all of it.
    """
    out = np.full(len(frames), -1, np.int8)
    if len(frames) < 2:
        return out
    still = (frames[1:] == frames[:-1]).mean(axis=(1, 2))
    shares = np.stack([_overlap(frames, dy, dx) for dy, dx in SHIFTS])
    best = shares.argmax(0)
    moved = (shares.max(0) >= match) & (still < match)
    out[1:][moved] = np.array(list(SHIFTS.values()), np.int8)[best[moved]]
    return out


def step_starts(directions: np.ndarray, still: int = 3) -> np.ndarray:
    """(m, 2) rows [frame, bit]: where a walk starts or turns.

    A start is a scrolling frame after at least `still` frames without
    scrolling (the player stood, so the game read the d-pad), or one whose
    direction differs from the last scrolling frame's.
    """
    rows = []
    last_dir, since = -1, still
    for t, d in enumerate(directions):
        if d < 0:
            since += 1
            continue
        if since >= still or d != last_dir:
            rows.append((t, int(d)))
        last_dir, since = int(d), 0
    return np.array(rows, np.int64).reshape(-1, 2)
