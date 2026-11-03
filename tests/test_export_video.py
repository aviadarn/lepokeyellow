import numpy as np

from poke.synth import BUTTONS
from scripts.export_video import choose, greys, segments

K = {name: k for k, name in enumerate(BUTTONS)}


def block(frames=24, **held):
    out = np.zeros((frames, 8), bool)
    for name, (a, b) in held.items():
        out[a:b, K[name]] = True
    return out


def test_a_tap_beats_a_walk_and_onsets_only():
    held = block(right=(0, 24), a=(10, 14))
    assert choose(held, held.astype(float), np.zeros(8, bool))[0] == "a"
    # A held from before the block has no onset in it: the walk wins.
    held = block(right=(0, 24), a=(0, 4))
    before = np.zeros(8, bool)
    before[K["a"]] = True
    assert choose(held, held.astype(float), before)[0] == "right"


def test_short_direction_taps_are_pass():
    held = block(up=(0, 5))
    assert choose(held, held.astype(float), np.zeros(8, bool))[0] is None


def test_segments_split_at_cuts_and_unreadable_frames():
    n = 1000
    cut = np.zeros(n)
    cut[300] = 0.9
    state = np.zeros((n, 8), np.uint8)
    state[600:650] = 255
    assert segments(cut, state, min_frames=100) == [(0, 300), (300, 600), (650, 1000)]


def test_greys_map_black_and_white_to_the_dmg_ends():
    out = greys(np.array([[24, 24, 24], [248, 248, 248]], np.uint8))
    assert out.tolist() == [0, 255]


def test_dreamer_export_is_the_same_read_in_small_batches(tmp_path, monkeypatch):
    import h5py

    import scripts.export_video as export

    rng = np.random.default_rng(0)
    n = 1000
    refs = np.repeat(np.arange(n // 2), 2).astype(np.int32)
    colours = rng.integers(0, 256, (42, 3), dtype=np.uint8)
    held = (rng.random((n, 8)) < 0.2).astype(np.float16)

    def run(batch, name):
        monkeypatch.setattr(export, "OBS_BATCH", batch)
        with h5py.File(tmp_path / f"{name}.h5", "w") as h5:
            h5["frame_ref"] = refs
            h5["colours"] = colours
            h5["uframes"] = np.broadcast_to(
                (np.arange(n // 2) % 42).astype(np.uint8)[:, None, None], (n // 2, 144, 160))
            h5["held_p"] = held
            export.export_dreamer(h5, h5, [(0, 600), (600, n)], tmp_path / name)
        with np.load(tmp_path / name / "transitions_0001.npz") as data:
            return {key: data[key] for key in data.files}

    big, small = run(512, "big"), run(5, "small")
    assert big.keys() == small.keys()
    for key in big:
        assert np.array_equal(big[key], small[key]), key
    assert big["obs"].shape[0] == 600 // 24 + 400 // 24


def test_repaired_beats_ruled_beats_decoded():
    from scripts.export_video import label_arrays

    n = 20
    up, down, left = 1, 2, 4
    labels = {"held_p": np.zeros((n, 8), np.float16),
              "buttons": np.full(n, up, np.uint8),
              "buttons_ruled": np.full(n, down, np.uint8),
              "buttons_repaired": np.full(n, left, np.uint8),
              "verified": np.r_[np.ones(10, bool), np.zeros(10, bool)],
              "rule_changed": np.zeros(n, bool), "rule_event": np.zeros(n, bool)}
    held, held_p, verified, changed, events, unexplained = label_arrays(labels)
    assert held[:10, 2].all() and not held[:10, 1].any()      # verified: repaired (left)
    assert held[10:, 1].all() and not held[10:, 0].any()      # elsewhere: ruled (down)
    assert (held_p[:10, 2] == 1).all()                         # verified frames are certain
    del labels["buttons_ruled"]
    held, *_ = label_arrays(labels)
    assert held[10:, 0].all()                                  # no ruled labels: decoded (up)


def test_export_carries_rule_counts(tmp_path):
    import h5py

    import scripts.export_video as export

    n = 240
    with h5py.File(tmp_path / "v.h5", "w") as h5:
        h5["frame_ref"] = np.arange(n, dtype=np.int32)
        h5["colours"] = np.zeros((42, 3), np.uint8)
        h5["uframes"] = np.zeros((n, 144, 160), np.uint8)
        h5["held_p"] = np.zeros((n, 8), np.float16)
        h5["buttons"] = np.zeros(n, np.uint8)
        events = np.zeros(n, bool)
        events[[5, 30, 31]] = True                               # step 0 has one, step 1 two
        changed = np.zeros(n, bool)
        changed[30:35] = True                                    # step 1: five frames fixed
        h5["rule_event"], h5["rule_changed"] = events, changed
        unexplained = np.zeros(n, bool)
        unexplained[30] = True                                   # one of step 1's events failed
        h5["rule_unexplained"] = unexplained
        export.export_dreamer(h5, h5, [(0, n)], tmp_path / "out")
    with np.load(tmp_path / "out" / "transitions_0001.npz") as d:
        assert d["rule_events"][:2].tolist() == [1, 2]
        assert d["rule_fixed"][:2].tolist() == [0, 5]
        assert d["rule_events"].sum() == 3 and d["rule_fixed"].sum() == 5
        assert d["rule_unexplained"][:2].tolist() == [0, 1]
