import numpy as np

from poke.label_rules import SHIFTS, enforce_walking, scroll, step_starts, walking_agreement


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


LEFT, UP, A = 1 << 2, 1 << 0, 1 << 4


def _walk_case(n=40):
    """A step left starts at frame 20; the labels hold nothing before it."""
    starts = np.array([[20, 2]])
    held_p = np.zeros((n, 8), np.float32)
    held_p[14:28, 2] = 0.3            # the model saw a left walk, weakly
    return np.zeros(n, np.int64), starts, held_p


def test_missing_step_press_is_inserted():
    masks, starts, held_p = _walk_case()
    masks[18] = UP | A                # a wrong direction and an A on a fixed frame
    assert walking_agreement(masks, starts) == 0.0
    out, changed = enforce_walking(masks, starts, held_p)
    assert out[18] == LEFT | A and out[19] == LEFT
    assert np.flatnonzero(changed).tolist() == [18, 19]
    assert walking_agreement(out, starts) == 1.0


def test_satisfied_start_is_left_alone():
    masks, starts, held_p = _walk_case()
    masks[17:30] = LEFT
    out, changed = enforce_walking(masks, starts, held_p)
    assert np.array_equal(out, masks) and not changed.any()


def test_rule_never_removes_a_press():
    masks, _, held_p = _walk_case()
    masks[5:9] = UP                   # a press into a wall: no step follows
    out, changed = enforce_walking(masks, np.zeros((0, 2), np.int64), held_p)
    assert np.array_equal(out, masks) and not changed.any()


def test_scripted_walk_gets_no_press():
    masks, starts, _ = _walk_case()
    out, changed = enforce_walking(masks, starts, np.zeros((40, 8), np.float32))
    assert np.array_equal(out, masks) and not changed.any()

