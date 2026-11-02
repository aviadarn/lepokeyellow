"""What the screen proves about the buttons, anywhere in a video.

The IDM's labels are a guess, and replaying them in the emulator only checks
the opening of the longplay (past it, edit cuts and the random number
generator part the two games). Two things the picture itself shows hold
everywhere, and each says a button was pressed:

  * walking. The overworld scrolls 2 px on every other frame while the
    player walks, 16 frames a step, in the walking direction; a step's
    direction is held just before it starts (true labels: 99.0 % of 301
    steps, within frames t-3 .. t-1 of the first scrolling frame t). But v3
    already gets 98 % of steps right and misses mostly forced movement, so
    this is a check, not a fixer (results/label_rules_synth.json);
  * text. A `▼` prompt waits for a new A or B press (true labels: 11 of 12
    waits answered without A were answered with B -- B does not advance
    Oak's intro speech, but it does answer ordinary text).

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


def step_starts(directions: np.ndarray, still: int = 3, window: int = 8,
                min_scrolls: int = 4) -> np.ndarray:
    """(m, 2) rows [frame, bit]: where a walk starts or turns.

    A start is a scrolling frame after at least `still` frames without
    scrolling (the player stood, so the game read the d-pad), or one whose
    direction differs from the last scrolling frame's -- and only if the
    next `window` frames scroll that way at least `min_scrolls` times and
    never another way. A step scrolls 2 px on every other frame for 16
    frames; on the video, single shifted-looking frames and a hit shaking
    the screen up and down were 18 % of "steps" before this.
    """
    rows = []
    last_dir, since = -1, still
    for t, d in enumerate(directions):
        if d < 0:
            since += 1
            continue
        if since >= still or d != last_dir:
            ahead = directions[t:t + window]
            if (ahead == d).sum() >= min_scrolls and ((ahead < 0) | (ahead == d)).all():
                rows.append((t, int(d)))
        last_dir, since = int(d), 0
    return np.array(rows, np.int64).reshape(-1, 2)


#: Frames around a step's first scrolling frame t in which its direction
#: must be held (t-3 .. t: the game reads the pad before moving, and the
#: capture emulator and PyBoy can disagree by a frame -- on the video the
#: model puts 67.5 % of step presses on t itself; true synthetic labels pass
#: either way), and the ones the rule holds it on when it is not.
WALK_LEAD = (0, 1, 2, 3)
WALK_HOLD = (1, 2)
DIRECTIONS = 0b1111


def _held_before(masks: np.ndarray, t: int, bit: int) -> bool:
    return any(t - k >= 0 and masks[t - k] >> bit & 1 for k in WALK_LEAD)


def walking_agreement(masks: np.ndarray, starts: np.ndarray) -> float:
    """Share of step starts whose direction the labels hold in t-3 .. t."""
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



#: The `▼` the game draws at tile row 16, column 18 (tile 0xEE in wTileMap)
#: while text waits for A, copied off the cartridge; it blinks by swapping
#: with a blank tile.
PROMPT_BOX = (slice(128, 136), slice(144, 152))
PROMPT_GLYPH = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]], bool)
#: The text box's rows; a redraw changes far more of them than a blink.
TEXT_ROWS = slice(104, 144)
ANSWER_CHANGE = 0.02
#: Frames a decoded tap lasts (scripts.label_video.TAP_FRAMES).
TAP = 5
A_BIT = 4
B_BIT = 5
#: The game waits a frame or two before the arrow is drawn, so a press that
#: early still answers it.
ANSWER_LEAD = 2


def prompt_visible(frames: np.ndarray, colours: np.ndarray) -> np.ndarray:
    """(n,) bool: the `▼` is on screen. `frames` are colour indices into
    `colours`; a pixel is dark when its RGB sums under 384, as on the
    cartridge, and the 8x8 box must match the glyph on 60 of 64 pixels."""
    # Index 255 marks a colour outside the table (clean synthetic frames);
    # it is never dark.
    dark = np.zeros(256, bool)
    dark[:len(colours)] = colours.astype(np.int32).sum(1) < 384
    box = dark[frames[:, PROMPT_BOX[0], PROMPT_BOX[1]]]
    return (box == PROMPT_GLYPH).sum(axis=(1, 2)) >= 60


def text_change(frames: np.ndarray) -> np.ndarray:
    """(n,) float32: share of the text box that differs from the frame
    before, the prompt's own tile left out; 0 for frame 0."""
    out = np.zeros(len(frames), np.float32)
    if len(frames) < 2:
        return out
    diff = frames[1:, TEXT_ROWS] != frames[:-1, TEXT_ROWS]
    rows = slice(PROMPT_BOX[0].start - TEXT_ROWS.start, PROMPT_BOX[0].stop - TEXT_ROWS.start)
    diff[:, rows, PROMPT_BOX[1]] = False
    area = diff.shape[1] * diff.shape[2] - 64
    out[1:] = diff.sum(axis=(1, 2)) / area
    return out


def answers(visible: np.ndarray, change: np.ndarray) -> np.ndarray:
    """(m, 2) rows [first frame the prompt shows, frame the text moves on].

    A wait starts at a visible prompt and ends at the first later frame
    where the box changes by more than ANSWER_CHANGE; the blinking glyph
    alone never does. A wait the video never answers is dropped.
    """
    rows, start = [], None
    for t in range(len(visible)):
        if start is None:
            if visible[t]:
                start = t
        elif change[t] > ANSWER_CHANGE:
            rows.append((start, t))
            start = None
    return np.array(rows, np.int64).reshape(-1, 2)


def _answered(masks: np.ndarray, lo: int, hi: int) -> bool:
    """A new A or B press in frames lo - ANSWER_LEAD .. hi."""
    for bit in (A_BIT, B_BIT):
        held = (masks >> bit) & 1
        if any(held[t] and (t == 0 or not held[t - 1])
               for t in range(max(lo - ANSWER_LEAD, 0), hi + 1)):
            return True
    return False


def text_agreement(masks: np.ndarray, spans: np.ndarray) -> float:
    """Share of answered prompts with a new A or B press while they waited."""
    if not len(spans):
        return 1.0
    return float(np.mean([_answered(masks, int(s), int(e)) for s, e in spans]))


def enforce_text(masks: np.ndarray, spans: np.ndarray,
                 onset_a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(masks, changed): every answered prompt has a new press in its wait.

    A wait without a new A or B gets a TAP-frame A press where the A onset head
    peaks inside it. If A is already held on the frame before (held to
    speed up the text), that one frame is released: Gen 1 answers the
    prompt on a new press only, so a hold through the wait cannot be right.
    """
    out = masks.copy()
    changed = np.zeros(len(masks), bool)
    bit = 1 << A_BIT
    for s, e in spans:
        s, e = int(s), int(e)
        if _answered(out, s, e):
            continue
        t = s + int(np.argmax(onset_a[s:e + 1]))
        if t > 0 and out[t - 1] & bit:
            out[t - 1] &= ~bit
            changed[t - 1] = True
        for f in range(t, min(t + TAP, len(out))):
            if not out[f] & bit:
                out[f] |= bit
                changed[f] = True
    return out, changed
