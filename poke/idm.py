"""Inverse dynamics model: which buttons were held, read off the frames.

Input is a window of consecutive frames as colour indices (poke/video.py
snaps the longplay to the same 42-colour set PyBoy draws, so synthetic and
video frames share one vocabulary). Each frame is seen together with the
one before it and a "pixel changed" plane, so a 2-pixel scroll is visible to
the first convolution. A small strided CNN embeds each frame pair, a
non-causal transformer reads the whole window -- the effect of a press shows
up frames later, and the model may look ahead -- and two heads come out per
frame:

  held      8 logits, one per button (poke.synth.BUTTONS order);
  mattered  8 logits: would flipping that button on this frame have changed
            the screen. Trained only where the exact counterfactual mask was
            computed, it says which `held` predictions are evidence and
            which are the prior speaking.
  onset     8 logits: was the button pressed ON this frame (released the
            frame before). A tap's first frame is what the game reads, and
            how long a player held A afterwards is mostly invisible; v1 read
            held-A at F1 ~0.2 and missed the press that opens the bedroom
            SNES text, which ended its replay of the real video.
  cut       1 logit: does this frame start a different piece of footage than
            the frame before it. The longplay had its grinding edited out;
            a frame-pair test cannot tell those cuts from the game's own
            jumps (menus closing, fades, palette flashes, Flash lighting a
            cave), so the model learns it from training windows that were
            spliced together from two places.
"""

from __future__ import annotations

import torch
from torch import nn

NUM_BUTTONS = 8
#: Colour vocabulary: the PyBoy CGB colours plus one "unknown" slot.
MAX_COLOURS = 64


class FrameEncoder(nn.Module):
    def __init__(self, embed: int = 4, width: int = 256):
        super().__init__()
        self.embed = nn.Embedding(MAX_COLOURS, embed)
        channels = 2 * embed + 1
        # Patchify first (4x4, stride 4): every pixel of a patch still has its
        # own weight, so a 2-pixel scroll is visible. Measured on the M5 at
        # batch 32 x 32 frames: 792 frames/s training with an 8-d embedding
        # and a stride-2 first layer, 2,057 like this (2,614 under bf16).
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 64, 4, stride=4), nn.GELU(),              # 36 x 40
            nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.GELU(),         # 18 x 20
            nn.Conv2d(96, 128, 3, stride=2, padding=1), nn.GELU(),        # 9 x 10
        )
        self.project = nn.Linear(128 * 9 * 10, width)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (B, T+1, H, W) long -> (B, T, width) for frames 1..T."""
        b, t1, h, w = frames.shape
        current, previous = frames[:, 1:], frames[:, :-1]
        changed = (current != previous).float().unsqueeze(2)
        e_cur = self.embed(current).permute(0, 1, 4, 2, 3)
        e_prev = self.embed(previous).permute(0, 1, 4, 2, 3)
        x = torch.cat([e_cur, e_prev, changed], dim=2).flatten(0, 1)
        x = self.conv(x).flatten(1)
        return self.project(x).view(b, t1 - 1, -1)


class IDM(nn.Module):
    def __init__(self, window: int = 32, width: int = 256, layers: int = 4, heads: int = 4):
        super().__init__()
        self.window = window
        self.encoder = FrameEncoder(width=width)
        self.position = nn.Parameter(torch.zeros(1, window, width))
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.held = nn.Linear(width, NUM_BUTTONS)
        self.mattered = nn.Linear(width, NUM_BUTTONS)
        self.onset = nn.Linear(width, NUM_BUTTONS)
        self.cut = nn.Linear(width, 1)

    def forward(self, frames: torch.Tensor):
        """frames: (B, window+1, H, W) colour indices; the first frame is
        context only. Returns (held, mattered, cut, onset) logits:
        (B, window, 8), (B, window, 8), (B, window), (B, window, 8)."""
        x = self.encoder(frames) + self.position[:, :frames.shape[1] - 1]
        x = self.norm(self.temporal(x))
        return self.held(x), self.mattered(x), self.cut(x).squeeze(-1), self.onset(x)


def onset_targets(held: torch.Tensor, previous: torch.Tensor):
    """(target, weight) for the onset head from held bits (B, T, 8) and the
    bits of the frame before the window (B, 8). Frames one away from an
    onset are not penalised either way: the evidence pins a press to the
    frame the game read it, and a press a frame early or late reads alike."""
    before = torch.cat([previous[:, None], held[:, :-1]], dim=1)
    onset = (held == 1) & (before == 0)
    near = torch.zeros_like(onset)
    near[:, 1:] |= onset[:, :-1]
    near[:, :-1] |= onset[:, 1:]
    weight = (~(near & ~onset)).float()
    return onset.float(), weight


def onset_loss(onset_logits, target, weight, pos_weight: float = 30.0):
    return nn.functional.binary_cross_entropy_with_logits(
        onset_logits, target, weight=weight,
        pos_weight=onset_logits.new_tensor(pos_weight))


def cut_loss(cut_logits, cut, pos_weight: float = 20.0):
    """BCE on the per-frame cut flag; cuts are one frame in a spliced window."""
    return nn.functional.binary_cross_entropy_with_logits(
        cut_logits, cut.float(), pos_weight=cut_logits.new_tensor(pos_weight))


def loss(held_logits, mattered_logits, held, mattered, pos_weight=None):
    """BCE on every held bit; BCE on mattered where it was computed (!= 255).

    `pos_weight` (8,) up-weights held=1 per button. Without it the first
    model never once predicted A, B, START or SELECT: A is a 5-frame tap on
    ~5 % of frames and only its first frame changes anything.
    """
    bce = nn.functional.binary_cross_entropy_with_logits
    held_loss = bce(held_logits, held.float(), pos_weight=pos_weight)
    known = mattered != 255
    if known.any():
        mattered_loss = bce(mattered_logits[known], mattered[known].float())
    else:
        mattered_loss = held_logits.new_zeros(())
    return held_loss, mattered_loss
