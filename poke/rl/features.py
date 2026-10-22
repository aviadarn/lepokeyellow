"""The world model, frozen, as the policy's eyes.

The encoder is the only part of the JEPA the policy uses: 144x160 greyscale in,
192 floats out. It is frozen during RL by the design -- one refresh is allowed
between data collections, never a gradient from the policy -- so this wrapper
puts the model in eval mode and clears requires_grad once, at load.

Only the newest frame is encoded each step. The policy sees four, and three of
them it saw last step already, so `EmbeddingStack` keeps them and the encoder
runs once per action instead of four times. At the measured ~1,758 frames/s
that is the difference between the encoder being the bottleneck and being four
times the bottleneck.

The geometry check is not ceremony. `scripts/train.py` takes --num-frames and
the model takes --dim, and a checkpoint trained with other values loads into
this class without complaint and produces a plausible, wrong observation. The
load raises instead, naming both numbers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from poke.goals import default_device
from poke.jepa import JEPA


def checkpoint_geometry(blob: dict) -> tuple[int, int | None]:
    """(embedding dim, frames per window) as the checkpoint recorded them.

    The dim is read from the weights rather than the config, because
    DefaultConfig has no dim field -- the encoder's class token carries it.
    """
    weights = blob["jepa"]
    dim = int(weights["encoder.cls"].shape[-1])
    config = blob.get("config") or {}
    frames = config.get("num_frames")
    return dim, None if frames is None else int(frames)


class FrozenEncoder:
    def __init__(self, checkpoint, device: str | None = None, expect_frames: int = 4,
                 half: bool = False):
        self.path = Path(checkpoint)
        blob = torch.load(self.path, map_location="cpu", weights_only=False)
        dim, frames = checkpoint_geometry(blob)
        if frames is not None and frames != expect_frames:
            raise ValueError(
                f"{self.path} was trained with num_frames={frames}; this policy "
                f"is built for num_frames={expect_frames}. Retrain the encoder "
                "or rebuild the policy -- the window has to be the same one."
            )
        self.dim = dim
        self.num_frames = expect_frames
        self.device = device or default_device()
        self.model = JEPA(num_frames=expect_frames, dim=dim)
        self.model.load_state_dict(blob["jepa"])
        self.model.eval().to(self.device)
        self.model.requires_grad_(False)
        # The encoder is frozen, so half precision costs only precision. What
        # it buys is the per-call latency that bounds the training loop: the
        # driver's cost per tick is one forward of this model, measured larger
        # than the four emulators it describes.
        self.half = half
        if half:
            self.model.half()

    @property
    def out_dim(self) -> int:
        """What the policy's observation gets from the pixels."""
        return self.num_frames * self.dim

    @torch.no_grad()
    def embed(self, frames: np.ndarray) -> np.ndarray:
        """[B, 144, 160] uint8 luminance -> [B, dim] float32.

        The scaling matches scripts/play.py's to_tensor and poke.dataset, so a
        frame means the same thing to this encoder as it did in training.
        """
        pixels = np.asarray(frames, dtype=np.float32) / 127.5 - 1.0
        batch = torch.from_numpy(pixels).unsqueeze(1).to(self.device)
        if self.half:
            batch = batch.half()
        out = self.model.encode(batch).float()
        return out.detach().to("cpu").numpy().astype(np.float32)


class EmbeddingStack:
    """The last `num_frames` embeddings, oldest first.

    `reset` fills every slot with the first embedding rather than zero-padding:
    a zero window is a state the encoder never produces, and the policy would
    spend its first three actions of every episode in one.
    """

    def __init__(self, num_frames: int = 4, dim: int = 192):
        self.num_frames = num_frames
        self.dim = dim
        self.buffer: np.ndarray | None = None

    def reset(self, embedding: np.ndarray) -> None:
        self.buffer = np.tile(np.asarray(embedding, dtype=np.float32), (self.num_frames, 1))

    def push(self, embedding: np.ndarray) -> None:
        if self.buffer is None:
            raise RuntimeError("EmbeddingStack.reset must run before push")
        self.buffer = np.roll(self.buffer, -1, axis=0)
        self.buffer[-1] = np.asarray(embedding, dtype=np.float32)

    def vector(self) -> np.ndarray:
        if self.buffer is None:
            raise RuntimeError("EmbeddingStack.reset must run before vector")
        return self.buffer.reshape(-1)
