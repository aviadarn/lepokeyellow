import numpy as np

from poke.label_rules import SHIFTS, scroll, step_starts


def _image(seed=0):
    return np.random.default_rng(seed).integers(0, 8, (144, 160)).astype(np.uint8)


def _shifted(img, dy, dx):
    """The next frame after a scroll: new[y, x] == old[y + dy, x + dx]."""
    out = np.random.default_rng(1).integers(0, 8, img.shape).astype(np.uint8)
    h, w = img.shape
    out[max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)] = \
        img[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)]
    return out


def test_scroll_reads_each_direction():
    img = _image()
    for (dy, dx), bit in SHIFTS.items():
        assert scroll(np.stack([img, _shifted(img, dy, dx)]))[1] == bit


def test_still_and_partial_motion_are_not_scroll():
    img = _image()
    sprite = img.copy()
    sprite[50:90, 60:100] = np.roll(img[50:90, 60:100], 2, axis=1)   # one sprite walks
    assert scroll(np.stack([img, img])).tolist() == [-1, -1]
    assert scroll(np.stack([img, sprite])).tolist() == [-1, -1]


def test_step_starts_after_still_frames_or_turns():
    dirs = np.array([-1] * 4 + [2] * 8 + [-1] + [2] * 8 + [0] * 8, np.int8)
    assert step_starts(dirs).tolist() == [[4, 2], [21, 0]]
