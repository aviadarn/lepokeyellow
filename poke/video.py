"""Reading the longplay: from an 800x720 YouTube stream to Game Boy frames.

Measured on YouTube aqr_YgqF2vM (format 298, h264, 59.94 fps):

  * 800x720 is exactly 5x the Game Boy's 160x144, nearest-neighbour, with
    the pixel grid aligned at the origin and no border. Each 5x5 block is
    one Game Boy pixel; H.264 blurs its edges, so a pixel is the mean of the
    block's inner 3x3.
  * The capture emulator applied Gambatte's GBC colour correction. With
    PyBoy's 5-bit colour c = rgb // 8, R = 26r + 4g + 2b, G = 24g + 8b,
    B = 6r + 4g + 22b, clamped to 1023 and >> 2. On the bedroom it predicts
    all four colours: (88,184,248) -> (110,200,210), seen (110,199,205);
    (184,136,248) -> (182,164,222), seen (176,167,218); white -> 248, seen
    249; black -> 24, seen 24.

So nothing about colour has to be learned. Synthetic frames go through
`correct` to look like the video, and video pixels are `snap`ped to the
nearest corrected colour of the colours PyBoy actually draws, which inverts
the correction and removes codec noise in one step.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator

import numpy as np

SCALE = 5
H, W = 144, 160
VIDEO_H, VIDEO_W = H * SCALE, W * SCALE

# Gambatte's colour-correction matrix, rows = output R, G, B over 5-bit r, g, b.
CORRECTION = np.array([[26, 4, 2],
                       [0, 24, 8],
                       [6, 4, 22]], dtype=np.int32)


def correct(rgb: np.ndarray) -> np.ndarray:
    """PyBoy CGB RGB (multiples of 8) -> what the capture emulator showed."""
    five = rgb.astype(np.int32) // 8
    out = np.minimum(five @ CORRECTION.T, 1023) >> 2
    return out.astype(np.uint8)


def downsample(frame: np.ndarray) -> np.ndarray:
    """(720, 800, 3) -> (144, 160, 3): mean of each 5x5 block's inner 3x3."""
    blocks = frame.reshape(H, SCALE, W, SCALE, 3)[:, 1:4, :, 1:4, :]
    return (blocks.sum(axis=(1, 3), dtype=np.uint16) // 9).astype(np.uint8)


#: The longplay is constant-rate: frame i has pts i * 1001 / 60000 exactly.
FRAME_SECONDS = 1001 / 60000


def read_frames(path: str, first: int = 0, count: int | None = None,
                batch: int = 256) -> Iterator[np.ndarray]:
    """Yield (n, 144, 160, 3) batches of downsampled frames `first`, `first+1`, ...

    Addressed by frame index, not time: seeking to half a frame before
    `first` makes it the first frame out, and `-frames:v` stops after
    exactly `count`. Seeking by seconds with -t let neighbouring segments
    share a frame (35,966 frames out of a 35,964-frame stretch).
    Every frame comes through (no fps filter); the game clock is recovered
    from them later.
    """
    seek = max(0.0, (first - 0.5) * FRAME_SECONDS)
    command = ["ffmpeg", "-loglevel", "error", "-ss", f"{seek:.6f}", "-i", path]
    if count is not None:
        command += ["-frames:v", str(count)]
    command += ["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:"]
    size = VIDEO_H * VIDEO_W * 3
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    try:
        while True:
            raw = process.stdout.read(size * batch)
            if not raw:
                break
            n = len(raw) // size
            frames = np.frombuffer(raw[:n * size], np.uint8).reshape(n, VIDEO_H, VIDEO_W, 3)
            yield np.stack([downsample(f) for f in frames])
    finally:
        process.stdout.close()
        process.wait()


class Snapper:
    """Map noisy video colours back to the PyBoy colours they came from.

    `colours` is the set PyBoy draws (collected from emulator frames). Each
    is pushed through `correct`, and a video pixel takes the PyBoy colour
    whose corrected value is nearest. The 15-bit lookup table makes it one
    index per pixel; distance is returned so a frame the palette does not
    explain (a fade, a cut to a non-game screen) can be flagged.
    """

    def __init__(self, colours: np.ndarray):
        self.colours = np.unique(np.asarray(colours, np.uint8).reshape(-1, 3), axis=0)
        corrected = correct(self.colours).astype(np.int32)
        # Quantise the video side to 5 bits per channel for the table; the
        # codec noise it loses is far below the gap between palette colours.
        grid = np.stack(np.meshgrid(*[np.arange(32) * 8 + 4] * 3, indexing="ij"), -1)
        grid = grid.reshape(-1, 1, 3)
        distances = np.abs(grid - corrected[None]).sum(-1)
        self.table = distances.argmin(1).astype(np.int32)
        self.table_distance = distances.min(1).astype(np.int32)

    def _keys(self, frames: np.ndarray) -> np.ndarray:
        q = frames.astype(np.int32) >> 3
        return (q[..., 0] << 10) | (q[..., 1] << 5) | q[..., 2]

    def snap(self, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(PyBoy-colour frames, per-pixel L1 distance in corrected space)."""
        index, distance = self.index(frames)
        return self.colours[index], distance

    def index(self, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(index into `colours` per pixel, as u1, and the L1 distance).

        Measured on a CRF-28 x264 round trip of 2,400 PyBoy frames (the
        video's own bitrate and noise level: 345 kbps, mean distance 11.3
        against 11.9 on the longplay), 0.11 % of pixels snap to the wrong
        colour. Not zero: nearly every frame has a few, at colour edges.
        """
        if len(self.colours) > 256:
            raise ValueError("more than 256 colours do not fit a u1 index")
        keys = self._keys(frames)
        return self.table[keys].astype(np.uint8), self.table_distance[keys]
