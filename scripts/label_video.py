"""Read the buttons off the longplay with a trained IDM: data/longplay/labels.h5.

Windows of WINDOW+1 frames slide with stride WINDOW/2; each contributes the
predictions for its middle half, so every frame is read with at least a
quarter-window of context on both sides (the start and end of the video
borrow from padding).

Output, one row per video frame (poke.synth.BUTTONS order):
  held_p      (N, 8) f2   P(button held during this frame)
  mattered_p  (N, 8) f2   P(the press or release would have shown on screen)
  cut_p       (N,) f2     P(this frame starts different footage: an edit cut)
  cut         (N,) bool   cut_p fired AND the picture jumped (see `confirm_cuts`)
  onset_p     (N, 8) f2   P(button pressed on this frame)
  buttons     (N,) u1     decoded held mask (bit k = BUTTONS[k]): directions from
                          held_p, taps from onset_p peaks (see `decode`)
  state       (N, 8) u1   0 released, 1 held, 2 don't-care (mattered_p < 0.5,
                          the screen cannot tell), 255 unknown (frame not
                          explained by the palette: title screen, fades)
attrs: checkpoint path and step, video sha1, thresholds.

  uv run python -m scripts.label_video --checkpoint checkpoints/idm/best.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from poke.idm import IDM

DONT_CARE, UNKNOWN = 2, 255
#: A frame with more than this share of off-palette pixels is not read.
BAD_SHARE = 0.10
#: An edit cut replaces the picture: fewer than CUT_SAME of the pixels stay
#: the same, and no colour-for-colour swap explains CUT_REMAP of them.
CUT_SAME, CUT_REMAP = 0.5, 0.9


def frames_for(h5, refs: np.ndarray) -> np.ndarray:
    """Colour-index frames for video frames whose uframes rows are `refs`."""
    low, high = int(refs.min()), int(refs.max())
    block = h5["uframes"][low:high + 1]
    return block[refs - low]


@torch.no_grad()
def predict(model, get_frames, n: int, device, batch: int = 64):
    """(held_p, mattered_p, cut_p, onset_p) -- (n, 8), (n, 8), (n,), (n, 8) f2.

    `get_frames(indices)` returns the colour-index frames at those indices
    (clipped to 0..n-1 by the caller here), shape (len, 144, 160). Windows of
    window+1 frames slide by half a window; each keeps its middle half.
    """
    window = model.window
    half, quarter = window // 2, window // 4
    held = np.zeros((n, 8), np.float16)
    mattered = np.zeros((n, 8), np.float16)
    cut = np.zeros(n, np.float16)
    onset = np.zeros((n, 8), np.float16)
    # Window k covers frames [s, s + window] with s = k * half - quarter - 1;
    # its outputs are frames s+1 .. s+window and the middle half is kept.
    starts = np.arange(-quarter - 1, n - quarter - 1, half)
    for b in range(0, len(starts), batch):
        chunk = starts[b:b + batch]
        index = np.clip(chunk[:, None] + np.arange(window + 1)[None], 0, n - 1)
        frames = get_frames(index.ravel()).reshape(len(chunk), window + 1, 144, 160)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "mps"):
            h, m, c, o = model(torch.from_numpy(np.ascontiguousarray(frames)).to(device).long())
        h = torch.sigmoid(h.float()).cpu().numpy()
        m = torch.sigmoid(m.float()).cpu().numpy()
        c = torch.sigmoid(c.float()).cpu().numpy()
        o = torch.sigmoid(o.float()).cpu().numpy()
        for row, start in enumerate(chunk):
            first = start + 1 + quarter
            lo, hi = max(first, 0), min(first + half, n)
            if lo >= hi:
                continue
            keep = slice(quarter + lo - first, quarter + hi - first)
            held[lo:hi] = h[row, keep]
            mattered[lo:hi] = m[row, keep]
            cut[lo:hi] = c[row, keep]
            onset[lo:hi] = o[row, keep]
    return held, mattered, cut, onset


#: Buttons decoded from onsets (a tap's first frame is what the game reads)
#: rather than from the held head; directions are walked, so held decides.
TAP_BUTTONS = (4, 5, 6, 7)  # a, b, start, select
TAP_FRAMES = 5   # median human tap in the synthetic generator
NMS = 3


def one_direction(held: np.ndarray) -> np.ndarray:
    """At most one direction per frame: where held runs overlap, the newest
    press wins.

    Gen 1 reads a single direction, by priority, so a few frames of up+left
    while the thumb rolls from up to left walk up -- in the longplay that
    broke the bedroom replay at video frame 7195. The synthetic players never
    hold two directions, so every overlap is a model error.
    """
    held = held.copy()
    dirs = held[:, :4]
    n = len(held)
    frames = np.arange(n)
    start = np.full((n, 4), -1)
    for k in range(4):
        d = dirs[:, k]
        onset = d & ~np.r_[False, d[:-1]]
        start[:, k] = np.where(d, np.maximum.accumulate(np.where(onset, frames, 0)), -1)
    both = dirs.sum(1) > 1
    keep = start[both].argmax(1)
    dirs[both] = False
    dirs[np.flatnonzero(both), keep] = True
    held[:, :4] = dirs
    return held


def decode(held_p: np.ndarray, onset_p: np.ndarray | None, single_direction: bool = True) -> np.ndarray:
    """Per-frame button masks (n,) from the heads.

    Directions: held_p > 0.5, at most one at a time (`one_direction`). Taps:
    every local maximum of onset_p above 0.5 (no stronger one within NMS
    frames) starts a TAP_FRAMES hold. Without an onset head (older
    checkpoints) everything comes from held_p.
    """
    held = held_p.astype(np.float32) > 0.5
    if single_direction:
        held = one_direction(held)
    if onset_p is not None:
        onset = onset_p.astype(np.float32)
        n = len(onset)
        for k in TAP_BUTTONS:
            held[:, k] = False
            p = onset[:, k]
            for t in np.flatnonzero(p > 0.5):
                lo, hi = max(0, t - NMS), min(n, t + NMS + 1)
                if p[t] >= p[lo:hi].max():
                    held[t:t + TAP_FRAMES, k] = True
    return (held * (1 << np.arange(8))).sum(1).astype(np.int64)


def remap_share(a: np.ndarray, b: np.ndarray) -> float:
    """Share of pixels explained by sending each colour of `a` to its most
    common colour at the same places in `b`: 1.0 for a pure palette change."""
    pairs = a.astype(np.int64).ravel() * 256 + b.ravel()
    counts = np.bincount(pairs, minlength=256 * 256).reshape(256, 256)
    return float(counts.max(1).sum() / a.size)


def confirm_cuts(cut_p: np.ndarray, h5, refs: np.ndarray) -> np.ndarray:
    """(n,) bool: frames where the cut head fired AND the picture really jumped.

    On the longplay the head alone fired 4,180 times in 6.9 h, mostly on
    the game's own palette fades (Oak's speech, doors) and on menus or text
    boxes opening. A fade keeps every pixel's place and swaps colours; a menu
    keeps most pixels. Each frame pair inside a run of cut_p > 0.5 is tested.
    """
    out = np.zeros(len(cut_p), bool)
    fired = cut_p.astype(np.float32) > 0.5
    starts = np.flatnonzero(fired & ~np.r_[False, fired[:-1]])
    for start in starts:
        stop = start
        while stop < len(fired) and fired[stop]:
            stop += 1
        for t in range(max(start, 1), stop):
            a, b = frames_for(h5, refs[[t - 1, t]])
            if (a == b).mean() < CUT_SAME and remap_share(a, b) < CUT_REMAP:
                out[t] = True
                break
    return out


def label(model, h5, device, limit: int | None = None):
    refs = h5["frame_ref"][:]
    n = len(refs) if limit is None else min(limit, len(refs))
    return predict(model, lambda index: frames_for(h5, refs[index]), n, device)


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu")
    model = IDM(window=checkpoint["window"]).to(device).eval()
    # A checkpoint from before the cut / onset heads loads without them; its
    # cut_p and onset_p are noise, and `has_onset` says not to decode from it.
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    checkpoint["has_onset"] = not any(key.startswith("onset.") for key in missing)
    if unexpected or any(not key.startswith(("cut.", "onset.")) for key in missing):
        raise RuntimeError(f"checkpoint mismatch: missing {missing}, unexpected {unexpected}")
    return model, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="checkpoints/idm/best.pt")
    parser.add_argument("--video", default="data/longplay/video.h5")
    parser.add_argument("--out", default="data/longplay/labels.h5")
    parser.add_argument("--limit", type=int, default=None, help="only the first N frames")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, checkpoint = load_model(args.checkpoint, device)

    started = time.time()
    with h5py.File(args.video) as h5:
        held, mattered, cut, onset = label(model, h5, device, limit=args.limit)
        n = len(held)
        cuts = confirm_cuts(cut, h5, h5["frame_ref"][:n])
        bad = h5["snap_bad"][:n].astype(np.float32) > BAD_SHARE
        sha1 = h5.attrs["video_sha1"]
    masks = decode(held, onset if checkpoint["has_onset"] else None)
    state = ((masks[:, None] >> np.arange(8)) & 1).astype(np.uint8)
    state[mattered < 0.5] = DONT_CARE
    state[bad] = UNKNOWN

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as h5:
        h5["held_p"] = held
        h5["mattered_p"] = mattered
        h5["cut_p"] = cut
        h5["cut"] = cuts
        h5["onset_p"] = onset
        h5["buttons"] = masks.astype(np.uint8)
        h5["state"] = state
        h5.attrs.update(checkpoint=str(args.checkpoint), step=int(checkpoint["step"]),
                        video_sha1=sha1, bad_share=BAD_SHARE,
                        buttons="up,down,left,right,a,b,start,select")
    decided = (state != DONT_CARE) & (state != UNKNOWN)
    print(f"labelled {n} frames in {time.time() - started:.0f}s; "
          f"held share {np.round((state == 1).mean(0), 3).tolist()}; "
          f"decided share {decided.mean():.3f}; unknown frames {int(bad.sum())}; "
          f"cuts {int(cuts.sum())} (head fired on {int(((cut[1:] > 0.5) & (cut[:-1] <= 0.5)).sum())} runs)",
          flush=True)


if __name__ == "__main__":
    main()
