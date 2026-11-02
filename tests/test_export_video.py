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
