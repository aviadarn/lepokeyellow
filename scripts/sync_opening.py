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
     and no 8x8 patch of text differs (`text_mismatch_near`). Both sides go
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
import hashlib
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


class VideoWindow:
    """Video frames [first, first + count), indexed by absolute video frame:
    an anchored run hours into the video reads only its own frames (from 0
    it would hold the whole prefix, ~30 GB at 6 h)."""

    def __init__(self, h5, first: int, count: int):
        self.first = first
        self.frames = video_frames(h5, first, count)

    def __len__(self) -> int:
        return self.first + len(self.frames)

    def __getitem__(self, v):
        if isinstance(v, slice):
            start, stop, _ = v.indices(len(self))
            if start < self.first:
                raise IndexError(f"video frame {start} is before the window ({self.first})")
            return self.frames[start - self.first:stop - self.first]
        if not self.first <= v < len(self):
            raise IndexError(f"video frame {v} is outside the window")
        return self.frames[v - self.first]


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


def _tile_sums(mask: np.ndarray) -> np.ndarray:
    """Per 8x8 tile of the screen's own grid (text is drawn on it)."""
    h, w = mask.shape
    k = TEXT_PATCH
    return mask.reshape(h // k, k, w // k, k).sum((1, 3), dtype=np.int32)


def text_mismatch(a: np.ndarray, b: np.ndarray, classes: np.ndarray) -> int:
    """Most pixels that differ in any 8x8 patch made only of paper and ink
    in both frames (any position, so scrolled text is covered too). Below
    TEXT_LIMIT it may return any smaller count: when fewer paper/ink pixels
    differ in the whole frame than that, no patch can reach it."""
    ca, cb = classes[a], classes[b]
    differ = (ca != cb) & (ca != 2) & (cb != 2)
    total = int(np.count_nonzero(differ))
    if total < TEXT_LIMIT:
        return total
    judged = _patch_sums((ca == 2) | (cb == 2)) == 0
    if not judged.any():
        return 0
    return int(_patch_sums(ca != cb)[judged].max())


def tile_aligned(pyboy) -> bool:
    """The background sits on the screen's 8x8 grid (SCX, SCY multiples of
    8): text drawn into it is on the tiles `text_mismatch_near` compares."""
    return pyboy.memory[0xFF43] % TEXT_PATCH == 0 and pyboy.memory[0xFF42] % TEXT_PATCH == 0


def text_mismatch_near(frame: np.ndarray, video, v: int, classes: np.ndarray,
                       slack: int = TEXT_SLACK, tiled: bool = True) -> int:
    """`text_mismatch` against video frame v, or, when that is TEXT_LIMIT or
    more, tile by tile: each 8x8 tile of the screen's grid that is paper and
    ink takes the fewest differing pixels against that tile in any video
    frame within `slack` where it is paper and ink too, and the worst tile
    is returned (tiles, not sliding patches: a patch straddling two letters
    printed in different batches matches no frame). Only `tiled`: with the
    screen scrolled off the grid a wrong letter splits across tiles, each
    under the limit, so there it is the least whole-frame mismatch.

    Patch by patch because the game copies its tile map to the screen a
    third at a time: at 2:32 the capture emulator shows a line of text in
    batches one letter apart from PyBoy's, so no single neighbour frame
    matches though every letter does. A wrong letter matches no frame.
    Neighbours are read nearest first, and the search ends once every patch
    has a match under TEXT_LIMIT (it may then return less than the true
    worst, never TEXT_LIMIT or more).
    """
    worst = text_mismatch(frame, video[v], classes)
    if worst < TEXT_LIMIT:
        return worst
    order = sorted(range(max(v - slack, 0), min(v + slack + 1, len(video))),
                   key=lambda u: (abs(u - v), u))
    if not tiled:
        for u in order:
            worst = min(worst, text_mismatch(frame, video[u], classes))
            if worst < TEXT_LIMIT:
                break
        return worst
    ce = classes[frame]
    mine = _tile_sums(ce == 2) == 0
    best = np.full(mine.shape, np.iinfo(np.int32).max, np.int64)
    for u in order:
        cu = classes[video[u]]
        both = mine & (_tile_sums(cu == 2) == 0)
        best = np.where(both, np.minimum(best, _tile_sums(ce != cu)), best)
        judged = best[mine & (best < np.iinfo(np.int32).max)]
        if judged.size and judged.max() < TEXT_LIMIT and (best[mine] < TEXT_LIMIT).all():
            return int(judged.max())
    judged = best[mine & (best < np.iinfo(np.int32).max)]
    return int(judged.max()) if judged.size else 0


def in_sync(share: float, text: int | None) -> bool:
    """`text` is None when the share already failed and text was not judged."""
    return share >= SYNC_SHARE and text is not None and text < TEXT_LIMIT


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


def writes_from_rows(rows: np.ndarray) -> dict:
    """State writes as stored ((M, 3): video frame, address, value) -> the
    Replayer's {video frame: ((address, value), ...)}."""
    out: dict[int, tuple] = {}
    for frame, address, value in np.asarray(rows, np.int64).reshape(-1, 3):
        out[int(frame)] = out.get(int(frame), ()) + ((int(address), int(value)),)
    return out


def rows_from_writes(writes: dict) -> np.ndarray:
    rows = [(frame, address, value) for frame, pairs in sorted(writes.items())
            for address, value in pairs]
    return np.array(rows, np.int32).reshape(-1, 3)


class Replayer:
    """Labelled buttons pressed into the emulator a frame at a time and
    compared with the video, tracking the frame offset (see `replay`).

    `save` / `load` capture the emulator AND the tracker, so a repair can
    rewind to just before a frame, change the buttons and play on.
    """

    def __init__(self, pyboy, snapper: Snapper, video: np.ndarray, masks: np.ndarray,
                 offset: int, slack: int = 2, patience: int = 3):
        self.pyboy, self.video, self.masks, self.snapper = pyboy, video, masks, snapper
        self.lut = emulator_lut(snapper)
        self.classes = text_classes(snapper.colours)
        self.pad = Pad(pyboy)
        self.slack, self.patience = slack, patience
        self.e, self.offset, self.streak, self.pending = 0, offset, 0, 0
        # Hidden state the video's emulator had and PyBoy did not (tile
        # animation counters, random bytes), written just before the tick
        # that draws that video frame: {video frame: ((address, value), ...)}.
        self.writes: dict[int, tuple] = {}

    def frame(self) -> np.ndarray:
        q = np.asarray(self.pyboy.screen.ndarray[:, :, :3]).astype(np.int32) >> 3
        return self.lut[(q[..., 0] << 10) | (q[..., 1] << 5) | q[..., 2]]

    def step(self):
        """Tick one frame: (video frame, match share and `text_mismatch_near`
        at the tracked offset -- None when the share already failed), or
        None when that frame has no video neighbours to compare with."""
        v = self.e + self.offset
        self.pad.set(int(self.masks[v]) if 0 <= v < len(self.masks) else 0)
        for address, value in self.writes.get(v, ()):
            self.pyboy.memory[address] = value
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
        if scores[0] < SYNC_SHARE:
            return v, scores[0], None
        return v, scores[0], text_mismatch_near(frame, self.video, v, self.classes,
                                                tiled=tile_aligned(self.pyboy))

    def state_key(self) -> bytes:
        """A digest of what decides the frames to come: VRAM, WRAM, OAM,
        HRAM, the held buttons and the offset tracker. Banked memory is seen
        through its current bank only, so two equal keys are near-certain,
        not proven, equal states -- a search may prune on it, a verdict may
        not."""
        m = self.pyboy.memory
        tracker = bytes([self.offset & 0xFF, self.streak & 0xFF, self.pending & 0xFF, self.pad.held])
        return hashlib.blake2b(bytes(m[0x8000:0xA000]) + bytes(m[0xC000:0xE000])
                               + bytes(m[0xFE00:0xFEA0]) + bytes(m[0xFF80:0xFFFF]) + tracker,
                               digest_size=16).digest()

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
           patience: int = 3, pyboy=None, writes: dict | None = None):
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
    replayer.writes = writes or {}
    shares, offsets, seen, texts = [], [], [], []
    for _ in range(frames):
        before = replayer.offset
        out = replayer.step()
        if out is None:
            continue
        shares.append(out[1])
        offsets.append(before)
        seen.append(out[0])
        texts.append(-1 if out[2] is None else out[2])
    if own:
        pyboy.stop(save=False)
    # Text not judged (the share failed) is -1.
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
                writes = (writes_from_rows(labels["state_writes"][:])
                          if "state_writes" in labels else {})
                labels = {"buttons": labels[args.buttons][:]}
                shares, offsets, seen, texts = replay(h5, labels, snapper, offset, args.frames,
                                                      pyboy=env.pyboy if env else None,
                                                      writes=writes)
            report["state_writes"] = int(sum(len(p) for p in writes.values()))
            if env:
                env.close()
            synced = (shares >= SYNC_SHARE) & (texts < TEXT_LIMIT)
            # Lost = out of sync for a full second; a single unmatched frame is
            # usually the offset catching up.
            lost = np.convolve(~synced, np.ones(60), "valid") >= 60
            first_lost = int(np.argmax(lost)) if lost.any() else len(shares)
            report.update(replayed=int(len(shares)), in_sync_share=float(synced.mean()),
                          pixel_share_only=float((shares >= SYNC_SHARE).mean()),
                          text_mismatch_frames=int(((shares >= SYNC_SHARE)
                                                    & (texts >= TEXT_LIMIT)).sum()),
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
