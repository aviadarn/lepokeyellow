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
        unexplained = h5["rule_unexplained"][:]
    # One walk, the cut not counted; walking is reported, never applied.
    assert report["steps"] == 1 and report["walking_agreement"] == 0.0
    assert not (ruled[15:40] & LEFT).any()
    # One answered wait with no press: an A tap where it starts.
    assert report["answers"] == 1 and report["text_agreement_before"] == 0.0
    assert report["text_agreement_after"] == 1.0
    assert np.flatnonzero(ruled & A).tolist() == [50, 51, 52, 53, 54]
    assert np.flatnonzero(changed).tolist() == [50, 51, 52, 53, 54]
    assert np.flatnonzero(events).tolist() == [21, 60]
    # The labels pressed nothing, so both events were unexplained before the rules.
    assert np.flatnonzero(unexplained).tolist() == [21, 60]
    assert report["unexplained"] == 2


def test_limit_and_apply_together_are_refused(tmp_path, monkeypatch):
    import pytest

    from scripts import check_labels

    video, labels = tmp_path / "video.h5", tmp_path / "labels.h5"
    _fake_video(video)
    _labels(labels)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results").mkdir()
    monkeypatch.setattr(sys, "argv", ["check_labels", "--video", str(video), "--labels", str(labels),
                                      "--apply", "--limit", "40", "--cache", str(tmp_path / "ev.npz")])
    with pytest.raises(SystemExit):
        check_labels.main()
    with h5py.File(labels) as h5:
        assert "buttons_ruled" not in h5              # nothing truncated was written


def _run(tmp_path, monkeypatch, cut=(), unreadable=()):
    from scripts import check_labels

    video, labels = tmp_path / "video.h5", tmp_path / "labels.h5"
    _fake_video(video)
    _labels(labels)
    with h5py.File(labels, "a") as h5:
        c = h5["cut"][:]
        c[:] = False
        c[list(cut)] = True
        h5["cut"][:] = c
        st = h5["state"][:]
        st[list(unreadable)] = 255
        h5["state"][:] = st
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results").mkdir(exist_ok=True)
    monkeypatch.setattr(sys, "argv", ["check_labels", "--video", str(video), "--labels", str(labels),
                                      "--name", "t", "--apply", "--cache", str(tmp_path / "ev.npz")])
    check_labels.main()
    with h5py.File(labels) as h5:
        return np.flatnonzero(h5["rule_event"][:]).tolist()


def test_a_cut_on_a_walks_first_frame_does_not_start_a_step_there(tmp_path, monkeypatch):
    assert 21 in _run(tmp_path, monkeypatch)
    assert 21 not in _run(tmp_path, monkeypatch, cut=[21])


def test_a_cut_or_unreadable_frame_inside_a_wait_drops_it(tmp_path, monkeypatch):
    assert 60 not in _run(tmp_path, monkeypatch, cut=[55])
    assert 60 not in _run(tmp_path, monkeypatch, unreadable=[52])


def test_screen_events_do_not_depend_on_the_chunk(tmp_path):
    from scripts.check_labels import screen_events

    _fake_video(tmp_path / "video.h5")
    with h5py.File(tmp_path / "video.h5") as video:
        whole = screen_events(video, chunk=4096)
        for chunk in (1, 2, 7, 21):
            parts = screen_events(video, chunk=chunk)
            assert all(np.array_equal(a, b) for a, b in zip(whole, parts)), chunk
