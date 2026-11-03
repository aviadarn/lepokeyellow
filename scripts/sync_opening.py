"""Do the labels reproduce the video? Replay them in the emulator and see.

The only check on the real footage that needs no human: boot PyBoy in colour
with blank cartridge RAM, line its frames up with the longplay's, then press
exactly what the IDM read off each video frame and watch for the first frame
the emulator draws differently from the video.

  1. Align. The intro (copyright, Game Freak, Pikachu) needs no input, so the
     offset between emulator frame e and video frame e + offset is found by
     matching it, and the number of intro frames that then agree exactly says
     whether one video frame really is one game frame.
  2. Replay. From power-on, before each tick, hold the buttons the labels
     give for the video frame that tick will be compared to.
  3. Score. A frame is in sync if at least SYNC_SHARE of its pixels match
     and no 8x8 patch of text differs (`text_mismatch`). Both sides go
     through the same path (Gambatte correction, snap), so an off-palette
     screen like the title compares like with like.

It cannot run far: the first random event (the rival battle's damage rolls,
NPCs that wander) depends on exact timing since power-on, including a boot
ROM that differs between PyBoy and the capture emulator. How far it gets,
and what it got through, is the result.

  uv run python -m scripts.sync_opening --frames 12000
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import h5py
import numpy as np
from pyboy import PyBoy

from poke.env import SRAM_BYTES
from poke.synth import Pad
from poke.video import Snapper, correct

ROM = "pokeyellow.gbc"
SYNC_SHARE = 0.98
# The share alone passed a player named K where the video says RED: a name,
# a menu cursor or one word is under 2 % of the screen. So text is judged on
# its own, in 8x8 patches (one font tile) that are all paper and ink in both
# frames -- sprites, the title's Pikachu and wandering NPCs are never all
# paper and ink, and are left to the share.
TEXT_PATCH = 8
TEXT_LIMIT = 6        # pixels of one patch that differ: a different letter or cursor
TEXT_SLACK = 12       # frames: text printed a few frames early or late is timing


def boot() -> PyBoy:
    return PyBoy(ROM, window="null", ram_file=io.BytesIO(bytes(SRAM_BYTES)))


def emulator_index(pyboy, snapper: Snapper) -> np.ndarray:
    rgb = np.asarray(pyboy.screen.ndarray[:, :, :3])
    return snapper.index(correct(rgb)[None])[0][0]


def video_frames(h5, first: int, count: int) -> np.ndarray:
    refs = h5["frame_ref"][first:first + count]
    block = h5["uframes"][int(refs.min()):int(refs.max()) + 1]
    return block[refs - refs.min()]


def match(a: np.ndarray, b: np.ndarray) -> float:
    return float((a == b).mean())


def text_classes(colours: np.ndarray) -> np.ndarray:
    """(256,) int8 per colour index: 0 paper (white), 1 ink (black or the
    near-blacks codec noise snaps it to), 2 anything else, 255 included."""
    colours = np.asarray(colours, np.int32)
    classes = np.full(256, 2, np.int8)
    classes[:len(colours)][colours.sum(1) >= 720] = 0
    classes[:len(colours)][(colours.max(1) <= 40) & (np.ptp(colours, 1) <= 16)] = 1
    return classes


def _patch_sums(mask: np.ndarray) -> np.ndarray:
    total = np.pad(mask.astype(np.int32).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    k = TEXT_PATCH
    return total[k:, k:] - total[:-k, k:] - total[k:, :-k] + total[:-k, :-k]


def text_mismatch(a: np.ndarray, b: np.ndarray, classes: np.ndarray) -> int:
    """Most pixels that differ in any 8x8 patch made only of paper and ink
    in both frames (any position, so scrolled text is covered too)."""
    ca, cb = classes[a], classes[b]
    judged = _patch_sums((ca == 2) | (cb == 2)) == 0
    if not judged.any():
        return 0
    return int(_patch_sums(ca != cb)[judged].max())


def text_mismatch_near(frame: np.ndarray, video: np.ndarray, v: int, classes: np.ndarray,
                       slack: int = TEXT_SLACK) -> int:
    """`text_mismatch` against video frame v, or, when that is TEXT_LIMIT or
    more, the least against any video frame within `slack`: a letter that
    printed early or late matches a neighbour, a wrong letter matches none."""
    worst = text_mismatch(frame, video[v], classes)
    if worst < TEXT_LIMIT:
        return worst
    near = range(max(v - slack, 0), min(v + slack + 1, len(video)))
    return min(text_mismatch(frame, video[u], classes) for u in near)


def in_sync(share: float, text: int) -> bool:
    return share >= SYNC_SHARE and text < TEXT_LIMIT


def align(h5, snapper, span: int = 2400, search: int = 900) -> tuple[int, list[float]]:
    """Offset such that emulator frame e matches video frame e + offset.

    The emulator runs the intro with no input; each video frame of the
    intro is matched against the emulator frames within `search` of it, on
    frames that are not blank (a blank frame matches every blank frame).
    """
    pyboy = boot()
    emulated = []
    for _ in range(span + search):
        pyboy.tick(1, True)
        emulated.append(emulator_index(pyboy, snapper))
    pyboy.stop(save=False)
    emulated = np.stack(emulated)
    video = video_frames(h5, 0, span)
    informative = [v for v in range(span) if len(np.unique(video[v])) > 2]
    best, best_offset = -1.0, 0
    for offset in range(-search, search):
        pairs = [(v - offset, v) for v in informative[::8] if 0 <= v - offset < len(emulated)]
        if len(pairs) < 20:
            continue
        score = np.mean([match(emulated[e], video[v]) for e, v in pairs])
        if score > best:
            best, best_offset = score, offset
    per_frame = [match(emulated[v - best_offset], video[v])
                 for v in informative if 0 <= v - best_offset < len(emulated)]
    return best_offset, per_frame


def anchored(state: str):
    """A colour save state dressed as the longplay: RED / BLUE, TEXT FAST."""
    from poke.env import PokeEnv
    from poke.symbols import load
    from poke.warp import TEXT_FAST, set_names, set_text_speed

    env = PokeEnv(ROM, state, color=True)
    env.reset()
    symbols = load()
    set_names(env, symbols)
    set_text_speed(env, symbols, TEXT_FAST)
    return env


def find_anchor(env, h5, snapper, first: int, last: int) -> tuple[int, float]:
    """The video frame in [first, last) that best matches the state's screen."""
    frame = emulator_index(env.pyboy, snapper)
    video = video_frames(h5, first, last - first)
    shares = [match(frame, v) for v in video]
    best = int(np.argmax(shares))
    return first + best, float(shares[best])


def release_carried(masks: np.ndarray, start: int) -> np.ndarray:
    """Masks with every press already under way at `start` released until
    the labels release it.

    A replay from a save state begins at video frame `start`; a press that
    began before it happened before the state existed, and pressing it now
    would be a new press the player never made. (v3 tapped A at 6,783, one
    frame before the bedroom state's 6,784; re-pressed, it broke the replay
    within 5 frames.)
    """
    out = masks.copy()
    for k in range(8):
        t = start
        while t < len(masks) and masks[t] >> k & 1 and masks[t - 1] >> k & 1:
            out[t] &= ~(1 << k)
            t += 1
    return out


def emulator_lut(snapper: Snapper) -> np.ndarray:
    """15-bit PyBoy colour key -> snapped colour index, for every colour at
    once: `emulator_index` as one table lookup per pixel."""
    five = np.stack(np.meshgrid(*[np.arange(32)] * 3, indexing="ij"), -1).reshape(-1, 3)
    corrected = correct((five * 8).astype(np.uint8)).astype(np.int32) >> 3
    return snapper.table[(corrected[:, 0] << 10) | (corrected[:, 1] << 5) | corrected[:, 2]].astype(np.uint8)


class Replayer:
    """Labelled buttons pressed into the emulator a frame at a time and
    compared with the video, tracking the frame offset (see `replay`).

    `save` / `load` capture the emulator AND the tracker, so a repair can
    rewind to just before a frame, change the buttons and play on.
    """

    def __init__(self, pyboy, snapper: Snapper, video: np.ndarray, masks: np.ndarray,
                 offset: int, slack: int = 2, patience: int = 3):
        self.pyboy, self.video, self.masks = pyboy, video, masks
        self.lut = emulator_lut(snapper)
        self.classes = text_classes(snapper.colours)
        self.pad = Pad(pyboy)
        self.slack, self.patience = slack, patience
        self.e, self.offset, self.streak, self.pending = 0, offset, 0, 0

    def frame(self) -> np.ndarray:
        q = np.asarray(self.pyboy.screen.ndarray[:, :, :3]).astype(np.int32) >> 3
        return self.lut[(q[..., 0] << 10) | (q[..., 1] << 5) | q[..., 2]]

    def step(self):
        """Tick one frame: (video frame, match share and `text_mismatch_near`
        at the tracked offset), or None when that frame has no video
        neighbours to compare with."""
        v = self.e + self.offset
        self.pad.set(int(self.masks[v]) if 0 <= v < len(self.masks) else 0)
        self.pyboy.tick(1, True)
        self.e += 1
        if not self.slack <= v < len(self.video) - self.slack:
            return None
        frame = self.frame()
        scores = {j: match(frame, self.video[v + j]) for j in range(-self.slack, self.slack + 1)}
        best = max(scores, key=lambda j: (scores[j], -abs(j)))
        if best != 0 and scores[best] > scores[0] and scores[best] >= SYNC_SHARE:
            self.streak = self.streak + 1 if best == self.pending else 1
            self.pending = best
            if self.streak >= self.patience:
                self.offset += best
                self.streak = 0
        else:
            self.streak = 0
        return v, scores[0], text_mismatch_near(frame, self.video, v, self.classes)

    def save(self) -> tuple:
        buffer = io.BytesIO()
        self.pyboy.save_state(buffer)
        return buffer.getvalue(), (self.e, self.offset, self.streak, self.pending, self.pad.held)

    def load(self, snapshot: tuple) -> None:
        blob, (self.e, self.offset, self.streak, self.pending, held) = snapshot
        self.pyboy.load_state(io.BytesIO(blob))
        # The save state does not carry PyBoy's input latch: re-press what
        # was held going into this frame before the next tick.
        self.pad.reset()
        self.pad.set(held)


def replay(h5, labels, snapper, offset: int, frames: int, slack: int = 2,
           patience: int = 3, pyboy=None):
    """Replay labelled buttons from power-on, tracking the frame offset.

    The intro showed the offset is not fixed: with no input at all, 8 % of
    intro frames match the emulator frame one before or after the linear
    mapping, in runs, as if the two emulators disagree on lag frames during
    heavy animation. So after every tick the emulator frame is compared with
    video frames v-slack..v+slack, and the offset moves when a neighbour
    has matched strictly better for `patience` frames in a row.

    Returns per compared frame: (match share at the tracked offset, offset
    used, video frame, text mismatch).
    """
    own = pyboy is None
    pyboy = boot() if own else pyboy
    if "buttons" in labels:
        masks = labels["buttons"][:].astype(np.int64)
    else:
        held = labels["held_p"][:].astype(np.float32) > 0.5
        masks = (held * (1 << np.arange(8))).sum(1).astype(np.int64)
    video = video_frames(h5, 0, frames + abs(offset) + slack + 1)
    if 0 < offset < len(masks):
        masks = release_carried(masks, offset)
    replayer = Replayer(pyboy, snapper, video, masks, offset, slack, patience)
    shares, offsets, seen, texts = [], [], [], []
    for _ in range(frames):
        before = replayer.offset
        out = replayer.step()
        if out is None:
            continue
        shares.append(out[1])
        offsets.append(before)
        seen.append(out[0])
        texts.append(out[2])
    if own:
        pyboy.stop(save=False)
    return np.array(shares), np.array(offsets), np.array(seen), np.array(texts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", default="data/longplay/video.h5")
    parser.add_argument("--labels", default="data/longplay/labels.h5")
    parser.add_argument("--frames", type=int, default=12_000)
    parser.add_argument("--buttons", default="buttons",
                        help="labels dataset to replay, e.g. buttons_repaired")
    parser.add_argument("--out", default="results/sync_opening.json")
    parser.add_argument("--anchor-state", default=None,
                        help="start from this colour state instead of power-on, e.g. "
                             "states/cgb/bedroom.state (skips the intro, which the IDM "
                             "has never seen)")
    parser.add_argument("--anchor-search", type=int, nargs=2, default=(6000, 7600),
                        help="video frames to look for the anchor state's screen in")
    args = parser.parse_args()

    with h5py.File(args.video) as h5:
        snapper = Snapper(h5["colours"][:])
        offset, intro = align(h5, snapper)
        intro = np.array(intro)
        print(f"offset {offset}: emulator frame e = video frame e + {offset}; "
              f"{(intro >= SYNC_SHARE).sum()}/{len(intro)} informative intro frames match "
              f"(median share {np.median(intro):.4f})", flush=True)
        report = {"offset": offset, "intro_frames": int(len(intro)),
                  "intro_in_sync": int((intro >= SYNC_SHARE).sum()),
                  "intro_median_share": float(np.median(intro))}
        if Path(args.labels).exists():
            env = None
            if args.anchor_state:
                env = anchored(args.anchor_state)
                anchor, share = find_anchor(env, h5, snapper, *args.anchor_search)
                # Emulator frame 0 (the state) is video frame `anchor`.
                offset = anchor + 1
                report.update(anchor_state=args.anchor_state, anchor_video_frame=anchor,
                              anchor_share=share)
                print(f"anchor: {args.anchor_state} matches video frame {anchor} "
                      f"({share:.4f})", flush=True)
            with h5py.File(args.labels) as labels:
                labels = {"buttons": labels[args.buttons][:]}
                shares, offsets, seen, texts = replay(h5, labels, snapper, offset, args.frames,
                                                      pyboy=env.pyboy if env else None)
            if env:
                env.close()
            synced = (shares >= SYNC_SHARE) & (texts < TEXT_LIMIT)
            # Lost = out of sync for a full second; a single unmatched frame is
            # usually the offset catching up.
            lost = np.convolve(~synced, np.ones(60), "valid") >= 60
            first_lost = int(np.argmax(lost)) if lost.any() else len(shares)
            report.update(replayed=int(len(shares)), in_sync_share=float(synced.mean()),
                          pixel_share_only=float((shares >= SYNC_SHARE).mean()),
                          text_mismatch_frames=int((texts >= TEXT_LIMIT).sum()),
                          frames_before_lost=first_lost,
                          lost_at_video_frame=int(seen[min(first_lost, len(seen) - 1)]),
                          offset_moves=int((np.diff(offsets) != 0).sum()),
                          final_offset=int(offsets[-1]))
            print(f"replay: {synced.mean():.3f} of {len(shares)} frames in sync; lost for "
                  f"good (60 frames off) after {first_lost} frames, at video frame "
                  f"{report['lost_at_video_frame']}; offset moved "
                  f"{report['offset_moves']} times, ended {offsets[-1]}", flush=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
