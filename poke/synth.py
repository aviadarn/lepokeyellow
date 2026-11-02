"""Per-frame synthetic play: what the inverse dynamics model learns from.

PokeDreamer-schema data moves in 24-tick steps; a human holds buttons for
any number of frames, and the longplay is one game frame per video frame
(measured: the walking scroll keeps Gen 1's 17-frame tile rhythm with no
repeated or dropped frames). So the IDM's training data is driven a frame at
a time: decisions come from the frontier policy, timings from human-like
press and gap distributions, and every frame records the 8 held buttons.

Two things make it look like the video rather than like PyBoy:
  * `degrade`: Gambatte's colour correction, 5x upscale, x264 at CRF 28
    (the longplay's own bitrate and noise level), decode, downsample and
    snap back to colour indices -- the exact path a video frame takes;
  * names: RED / BLUE / PIKACHU, as in the longplay.

`mattered` is the honest ground truth for "could anyone have seen this
press": flip one button on one frame, replay the same future, compare the
screens. It is exact and slow (9 replays per frame), so it is computed on a
sample of frames, and everything else is labelled by the held bits alone.
"""

from __future__ import annotations

import io
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from poke.dreamer_policies import A, B, PASS, START, Frontier
from poke.ram import coords, cur_map, dialog_open, in_battle
from poke.video import SCALE, Snapper, correct, read_frames

#: Per-frame button order: the repo's six (poke.actions.BUTTON_ORDER), then
#: start and select, so `[:, :6]` slices back to the repo's convention.
BUTTONS = ("up", "down", "left", "right", "a", "b", "start", "select")
BIT = {name: 1 << k for k, name in enumerate(BUTTONS)}
#: PokeDreamer action index -> button name (poke.dreamer.ACTIONS order).
DREAMER_TO_BUTTON = {0: "down", 1: "left", 2: "right", 3: "up", 4: "a", 5: "b",
                     6: "start"}

CHECKPOINTS = (1, 4, 12, 32)
CRF = 28


def _lognormal(rng, median, sigma, low, high) -> int:
    return int(np.clip(round(rng.lognormal(np.log(median), sigma)), low, high))


class HumanLike:
    """Frontier's choices at human timings, one held-button mask per frame."""

    def __init__(self, rng: np.random.Generator, start_rate: float = 0.02, bump_a: float = 0.0):
        self.rng = rng
        self.start_rate = start_rate
        self.bump_a = bump_a
        self.frontier = Frontier(rng, epsilon=0.1)
        self.mask = 0
        self.hold = 0
        self.gap = 0
        self.walked_from = None

    def __call__(self, observe) -> int:
        if self.hold > 0:
            self.hold -= 1
            return self.mask
        if self.gap > 0:
            self.gap -= 1
            return 0
        fields = observe()
        free = not fields["in_battles"] and not fields["dialog_opens"]
        here = (fields["map_ids"], fields["ys"], fields["xs"])
        bumped = self.walked_from == here
        self.walked_from = None
        action = self.frontier(fields)
        rng = self.rng
        # Players open the START menu now and then (items, party, save). The
        # frontier policy only reaches it through its random moves, which
        # left START held on 0.03 % of frames -- too few for the model to
        # ever learn it.
        if free and rng.random() < self.start_rate:
            action = START
        # A walk that went nowhere ran into something: a person, a sign, a
        # PC, a bookshelf. Players press A there. Without this, 360k frames
        # of synthetic play held 19 A-opened text boxes, and the model missed
        # the longplay's A at the bedroom PC.
        elif self.bump_a and free and bumped and rng.random() < self.bump_a:
            action = A
        if action == PASS:
            self.mask, self.hold, self.gap = 0, 0, _lognormal(rng, 10, 0.8, 1, 90)
            return 0
        name = DREAMER_TO_BUTTON[action]
        if rng.random() < 0.005:
            name = "select"
        mask = BIT[name]
        if free and action not in (A, B, START):
            # Walking: holds from a turn-in-place tap to a long stride, and
            # most of the time straight into the next direction.
            hold = _lognormal(rng, 20, 0.9, 1, 300)
            gap = 0 if rng.random() < 0.6 else _lognormal(rng, 8, 0.8, 1, 60)
            if rng.random() < 0.03:
                mask |= BIT[rng.choice(["a", "b"])]
            self.walked_from = here
        else:
            # Taps: text, menus, battles, and A/B/START in the overworld.
            hold = _lognormal(rng, 5, 0.5, 1, 20)
            gap = _lognormal(rng, 12 if free else 14, 0.9, 1, 180)
        self.mask, self.hold, self.gap = mask, hold - 1, gap
        return mask


class MenuTapper:
    """Taps for screens with no overworld: intro, title, menus, naming.

    From power-on the frontier policy has nothing to walk on, and the
    longplay's first two minutes are exactly these screens (the player opens
    OPTION, types RED on the naming keyboard). Human-ish timing, as HumanLike.
    """

    BUTTONS = ("a", "start", "b", "up", "down", "left", "right")
    WEIGHTS = (0.40, 0.10, 0.10, 0.10, 0.12, 0.09, 0.09)

    def __init__(self, rng: np.random.Generator, idle: tuple[int, int] = (0, 120)):
        self.rng = rng
        self.mask = 0
        self.hold = 0
        # Frames before the first tap. The longplay's player let the whole
        # intro run (no input until the title screen, ~2,000 frames); a
        # window of (0, 120) skips it, and a model trained only on that
        # invented presses during the real intro.
        self.gap = int(rng.integers(*idle))

    def __call__(self, observe) -> int:
        if self.hold > 0:
            self.hold -= 1
            return self.mask
        if self.gap > 0:
            self.gap -= 1
            return 0
        rng = self.rng
        name = self.BUTTONS[rng.choice(len(self.BUTTONS), p=self.WEIGHTS)]
        self.mask = BIT[name]
        self.hold = _lognormal(rng, 5, 0.5, 1, 20) - 1
        self.gap = _lognormal(rng, 14, 0.9, 1, 180)
        return self.mask


class Pad:
    """Press and release PyBoy buttons so the held set equals a mask."""

    def __init__(self, pyboy):
        self.pyboy = pyboy
        self.held = 0

    def set(self, mask: int) -> None:
        changed = mask ^ self.held
        for k, name in enumerate(BUTTONS):
            if changed >> k & 1:
                (self.pyboy.button_press if mask >> k & 1 else self.pyboy.button_release)(name)
        self.held = mask

    def reset(self) -> None:
        for name in BUTTONS:
            self.pyboy.button_release(name)
        self.held = 0


def observe_fields(env) -> dict:
    y, x = coords(env)
    return {"map_ids": cur_map(env) or 0, "ys": y or 0, "xs": x or 0,
            "in_battles": in_battle(env), "dialog_opens": dialog_open(env)}


def play(env, frames: int, rng: np.random.Generator, sample_every: int = 20, policy=None):
    """Drive `frames` frames. Returns (rgb, masks, labels, blobs).

    rgb[t] is the screen after frame t, masks[t] the buttons held during it.
    blobs holds a save state from just before frame t for every sampled t
    (one in `sample_every`, at a random phase), for `mattered`.
    """
    pyboy = env.pyboy
    pad = Pad(pyboy)
    policy = policy or HumanLike(rng)
    rgb = np.empty((frames, 144, 160, 3), np.uint8)
    masks = np.empty(frames, np.uint8)
    labels = {key: np.empty(frames, np.uint8) for key in ("map_ids", "ys", "xs")}
    labels.update(in_battles=np.empty(frames, bool), dialog_opens=np.empty(frames, bool))
    blobs: dict[int, bytes] = {}
    phase = int(rng.integers(0, sample_every))
    for t in range(frames):
        if t % sample_every == phase:
            buffer = io.BytesIO()
            pyboy.save_state(buffer)
            blobs[t] = buffer.getvalue()
        mask = policy(lambda: observe_fields(env))
        pad.set(mask)
        pyboy.tick(1, True)
        rgb[t] = pyboy.screen.ndarray[:, :, :3]
        masks[t] = mask
        fields = observe_fields(env)
        for key in labels:
            labels[key][t] = fields[key]
    pad.reset()
    return rgb, masks, labels, blobs


def mattered(env, blob: bytes, future: np.ndarray, held_before: int) -> np.ndarray:
    """(8,) bool: did flipping button k on the first frame of `future` change
    the screen at any checkpoint, against the unflipped replay from the same
    state? Both runs restore the same blob, so restore effects cancel."""
    pyboy = env.pyboy
    pad = Pad(pyboy)

    def run(masks):
        pyboy.load_state(io.BytesIO(blob))
        pad.reset()
        # The save state does not carry PyBoy's input latch; re-press what
        # was held going into this frame before the first tick.
        pad.set(held_before)
        shots = []
        for k, mask in enumerate(masks, 1):
            pad.set(int(mask))
            pyboy.tick(1, k in CHECKPOINTS)
            if k in CHECKPOINTS:
                shots.append(pyboy.screen.ndarray[:, :, :3].tobytes())
        return shots

    span = future[:max(CHECKPOINTS)]
    base = run(span)
    out = np.zeros(len(BUTTONS), bool)
    for k in range(len(BUTTONS)):
        flipped = span.copy()
        flipped[0] ^= 1 << k
        out[k] = run(flipped) != base
    pad.reset()
    return out


def degrade(rgb: np.ndarray, snapper: Snapper, crf: int = CRF,
            chunk: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """PyBoy frames -> what the same frames look like after the longplay's
    path: (colour indices, per-frame mean snap distance).

    Streamed, `chunk` frames at a time, through a temporary file. The first
    version built the whole episode at 5x in memory -- 6,000 frames x
    800x720x3 = 10.4 GB per copy, about three copies alive, per worker --
    and with 8 workers it froze a 34 GB Mac (forced reset 2026-10-27,
    five minutes into the intro episodes). Peak memory is now a few chunks.
    """
    size = f"{160 * SCALE}x{144 * SCALE}"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "episode.mp4"
        encoder = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", size, "-r", "60000/1001", "-i", "pipe:",
             "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
             str(path)], stdin=subprocess.PIPE)
        try:
            for start in range(0, len(rgb), chunk):
                big = correct(rgb[start:start + chunk]).repeat(SCALE, 1).repeat(SCALE, 2)
                encoder.stdin.write(big.tobytes())
        finally:
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError(f"x264 encode failed ({encoder.returncode})")
        index, distance = [], []
        for batch in read_frames(str(path), batch=chunk):
            i, d = snapper.index(batch)
            index.append(i)
            distance.append(d.mean(axis=(1, 2)))
    index = np.concatenate(index)
    if len(index) != len(rgb):
        raise RuntimeError(f"decoded {len(index)} frames from {len(rgb)}")
    return index, np.concatenate(distance).astype(np.float16)


def clean_index(rgb: np.ndarray, colours: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Exact colour indices of PyBoy frames (255 for a colour not in the set).

    PyBoy's CGB colours are multiples of 8, so a 15-bit key is exact; a
    lookup table in chunks replaces an np.unique over every pixel of the
    episode (~2-3 GB per worker at 6,000 frames).
    """
    table = np.full(1 << 15, 255, np.uint8)
    five = colours.astype(np.int32) >> 3
    table[(five[:, 0] << 10) | (five[:, 1] << 5) | five[:, 2]] = np.arange(len(colours))
    out = np.empty(rgb.shape[:-1], np.uint8)
    for start in range(0, len(rgb), chunk):
        block = rgb[start:start + chunk]
        q = block.astype(np.int32)
        keys = ((q[..., 0] >> 3) << 10) | ((q[..., 1] >> 3) << 5) | (q[..., 2] >> 3)
        # A colour that is not a multiple of 8 is not one of PyBoy's: unknown.
        off_grid = ((q & 7) != 0).any(-1)
        result = table[keys]
        result[off_grid] = 255
        out[start:start + chunk] = result
    return out
