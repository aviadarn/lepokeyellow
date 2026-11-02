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


#: Frames before a step's first scrolling frame in which its direction must
#: be held (t-3 .. t-1), and the ones the rule holds it on when it is not.
WALK_LEAD = (1, 2, 3)
WALK_HOLD = (1, 2)
DIRECTIONS = 0b1111


def _held_before(masks: np.ndarray, t: int, bit: int) -> bool:
    return any(t - k >= 0 and masks[t - k] >> bit & 1 for k in WALK_LEAD)


def walking_agreement(masks: np.ndarray, starts: np.ndarray) -> float:
    """Share of step starts whose direction the labels hold in t-3 .. t-1."""
    if not len(starts):
        return 1.0
    return float(np.mean([_held_before(masks, int(t), int(bit)) for t, bit in starts]))


def enforce_walking(masks: np.ndarray, starts: np.ndarray, held_p: np.ndarray,
                    floor: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """(masks, changed): every step start's direction held just before it.

    A start the labels miss gets its direction on t-2 and t-1, the other
    directions cleared there (Gen 1 walks one way at a time). Nothing is
    removed anywhere else: a press into a wall scrolls nothing and is real.
    Where the model saw no sign of that direction around the step
    (held_p < `floor` over t-8 .. t), the player is being walked by a
    script, and no press is invented.
    """
    out = masks.copy()
    changed = np.zeros(len(masks), bool)
    for t, bit in starts:
        t, bit = int(t), int(bit)
        if _held_before(out, t, bit) or held_p[max(t - 8, 0):t + 1, bit].max() < floor:
            continue
        for k in WALK_HOLD:
            f = t - k
            if f < 0:
                continue
            new = (out[f] & ~DIRECTIONS) | (1 << bit)
            changed[f] |= new != out[f]
            out[f] = new
    return out, changed
