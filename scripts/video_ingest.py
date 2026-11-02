"""Decode the longplay into Game Boy frames: data/longplay/video.h5.

Each 5x5 block becomes one pixel (inner 3x3 mean), each pixel snaps to the
nearest of the colours PyBoy draws in CGB mode after Gambatte's correction
(poke/video.py), and runs of identical snapped frames are stored once.

  uv run python -m scripts.video_ingest --workers 8

Output:
  uframes    (U, 144, 160) u1   colour indices into `colours`
  colours    (C, 3) u1          PyBoy CGB RGB
  frame_ref  (N,) i4            video frame i (pts i*1001/60000) -> row of uframes
  snap_mean  (N,) f2            mean per-pixel snap distance
  snap_bad   (N,) f2            share of pixels further than BAD_DISTANCE
attrs: video id, file sha1, stream rate, colour set source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np

VIDEO_ID = "aqr_YgqF2vM"
RATE = (60000, 1001)
#: A pixel this far from every corrected colour is not explained by the
#: palette: a fade, a screen PyBoy data never showed, or an edit.
BAD_DISTANCE = 40


def frame_count(path: str) -> int:
    """Exact, from the container's packet count (constant-rate stream)."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                          "-count_packets", "-show_entries", "stream=nb_read_packets",
                          "-of", "json", path], capture_output=True, text=True, check=True)
    return int(json.loads(out.stdout)["streams"][0]["nb_read_packets"])


def _segment(job):
    path, first, count, colours, out = job
    from poke.video import Snapper, read_frames

    snapper = Snapper(colours)
    unique, refs, means, bads = [], [], [], []
    previous = None
    for batch in read_frames(path, first, count):
        index, distance = snapper.index(batch)
        means.append(distance.mean(axis=(1, 2)))
        bads.append((distance > BAD_DISTANCE).mean(axis=(1, 2)))
        for frame in index:
            if previous is None or not np.array_equal(frame, previous):
                unique.append(frame)
                previous = frame
            refs.append(len(unique) - 1)
    if len(refs) != count:
        raise RuntimeError(f"segment at frame {first}: decoded {len(refs)}, wanted {count}")
    np.savez(out, uframes=np.stack(unique), frame_ref=np.array(refs, np.int32),
             snap_mean=np.concatenate(means).astype(np.float16),
             snap_bad=np.concatenate(bads).astype(np.float16))
    return out, len(refs), len(unique)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", default=f"data/video/{VIDEO_ID}.mp4")
    parser.add_argument("--colours", default="data/colours_cgb.npy")
    parser.add_argument("--out", default="data/longplay/video.h5")
    parser.add_argument("--segment", type=int, default=18_000, help="frames per job")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="only the first N frames")
    args = parser.parse_args()

    out = Path(args.out)
    parts = out.parent / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    colours = np.load(args.colours)
    total = args.limit or frame_count(args.video)
    starts = range(0, total, args.segment)
    jobs = [(args.video, s, min(args.segment, total - s), colours,
             str(parts / f"seg_{k:04d}.npz")) for k, s in enumerate(starts)]
    print(f"{len(jobs)} segments of {args.segment} frames over {total} frames", flush=True)

    started = time.time()
    with Pool(args.workers) as pool:
        results = pool.map(_segment, jobs)
    frames = sum(n for _, n, _ in results)
    print(f"decoded {frames} frames in {time.time() - started:.0f}s", flush=True)

    sha1 = hashlib.sha1()
    with open(args.video, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            sha1.update(block)

    with h5py.File(out, "w") as h5:
        uniques = sum(u for _, _, u in results)
        uframes = h5.create_dataset("uframes", (uniques, 144, 160), np.uint8,
                                    chunks=(64, 144, 160), compression="lzf")
        refs = h5.create_dataset("frame_ref", (frames,), np.int32)
        mean = h5.create_dataset("snap_mean", (frames,), np.float16)
        bad = h5.create_dataset("snap_bad", (frames,), np.float16)
        h5["colours"] = colours
        u = n = 0
        for path, count, unique in results:
            with np.load(path) as part:
                uframes[u:u + unique] = part["uframes"]
                refs[n:n + count] = part["frame_ref"] + u
                mean[n:n + count] = part["snap_mean"]
                bad[n:n + count] = part["snap_bad"]
            u += unique
            n += count
        h5.attrs.update(video_id=VIDEO_ID, video_sha1=sha1.hexdigest(),
                        rate_num=RATE[0], rate_den=RATE[1],
                        colour_correction="gambatte", bad_distance=BAD_DISTANCE,
                        segment_frames=args.segment)
    print(f"wrote {out}: {frames} frames, {uniques} unique "
          f"({uniques / frames:.1%}), {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
