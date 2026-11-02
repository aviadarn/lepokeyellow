import json
import sys

import h5py
import numpy as np

from poke.label_rules import PROMPT_BOX, PROMPT_GLYPH

N = 80
LEFT, A = 1 << 2, 1 << 4


def _fake_video(path):
    """Frames 21-35 walk left (2 px on every other frame), a cut at 40 that
    looks like a scroll, a ▼ waiting on frames 50-59 and answered at 60."""
    rng = np.random.default_rng(0)
    frame = rng.choice([0, 2], (144, 160)).astype(np.uint8)   # white / blue texture
    frames = []
    for t in range(N):
        if 21 <= t <= 35 and t % 2 == 1:
            frame = np.concatenate([rng.choice([0, 2], (144, 2)).astype(np.uint8), frame[:, :-2]], 1)
        if t == 40:
            frame = np.concatenate([rng.choice([0, 2], (144, 2)).astype(np.uint8), frame[:, :-2]], 1)
        f = frame.copy()
        if t >= 45:
            f[104:144] = 0                                       # an empty text box
        if 50 <= t < 60:
            f[PROMPT_BOX] = PROMPT_GLYPH.astype(np.uint8)        # black arrow
        if t >= 60:
            f[104:144, 8:150] = 1                                # the next page of text
        frames.append(f)
    with h5py.File(path, "w") as h5:
        h5["uframes"] = np.stack(frames)
        h5["frame_ref"] = np.arange(N, dtype=np.int32)
        h5["colours"] = np.array([[248, 248, 248], [24, 24, 24], [120, 160, 220]], np.uint8)
        h5.attrs["video_sha1"] = "fake"


def _labels(path):
    with h5py.File(path, "w") as h5:
        h5["buttons"] = np.zeros(N, np.uint8)
        h5["held_p"] = np.full((N, 8), 0.2, np.float16)
        h5["onset_p"] = np.zeros((N, 8), np.float16)
        cut = np.zeros(N, bool)
        cut[40] = True
        h5["cut"] = cut
        h5["state"] = np.zeros((N, 8), np.uint8)


def test_check_labels_end_to_end(tmp_path, monkeypatch):
    from scripts import check_labels

    video, labels = tmp_path / "video.h5", tmp_path / "labels.h5"
    _fake_video(video)
    _labels(labels)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results").mkdir()
    monkeypatch.setattr(sys, "argv", ["check_labels", "--video", str(video), "--labels", str(labels),
                                      "--name", "t", "--apply", "--cache", str(tmp_path / "ev.npz")])
    check_labels.main()
    report = json.loads((tmp_path / "results" / "label_consistency_t.json").read_text())
    with h5py.File(labels) as h5:
        ruled, changed, events = h5["buttons_ruled"][:], h5["rule_changed"][:], h5["rule_event"][:]
    # One walk, the cut not counted; walking is reported, never applied.
    assert report["steps"] == 1 and report["walking_agreement"] == 0.0
    assert not (ruled[15:40] & LEFT).any()
    # One answered wait with no press: an A tap where it starts.
    assert report["answers"] == 1 and report["text_agreement_before"] == 0.0
    assert report["text_agreement_after"] == 1.0
    assert np.flatnonzero(ruled & A).tolist() == [50, 51, 52, 53, 54]
    assert np.flatnonzero(changed).tolist() == [50, 51, 52, 53, 54]
    assert np.flatnonzero(events).tolist() == [21, 60]
