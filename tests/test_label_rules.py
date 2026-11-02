import numpy as np

from poke.label_rules import (PROMPT_BOX, PROMPT_GLYPH, SHIFTS, TAP, answers, enforce_text,
                              enforce_walking, prompt_visible, scroll, step_starts, text_agreement,
                              text_change, walking_agreement)


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



COLOURS = np.array([[248, 248, 248], [24, 24, 24], [120, 160, 220]], np.uint8)   # white, black, blue


def _text_frame():
    frame = np.zeros((144, 160), np.uint8)              # all white
    frame[104:144, 8:120:2] = 1                          # some printed text
    return frame


def test_prompt_is_found_on_its_tile():
    plain = _text_frame()
    waiting = plain.copy()
    waiting[PROMPT_BOX] = PROMPT_GLYPH.astype(np.uint8)  # black where the glyph is
    assert prompt_visible(np.stack([plain, waiting]), COLOURS).tolist() == [False, True]


def test_text_change_ignores_the_prompt_tile():
    a = _text_frame()
    b = a.copy()
    b[PROMPT_BOX] = PROMPT_GLYPH.astype(np.uint8)        # only the arrow blinks on
    c = a.copy()
    c[104:144, :] = 2                                    # the box is redrawn
    change = text_change(np.stack([a, b, c]))
    assert change[0] == 0 and change[1] == 0 and change[2] > 0.5


def test_a_blinking_prompt_is_one_wait():
    visible = np.array([0] * 5 + [1] * 10 + [0] * 20 + [1] * 10 + [0] * 5, bool)
    change = np.zeros(len(visible), np.float32)
    change[47] = 0.10
    assert answers(visible, change).tolist() == [[5, 47]]


A_BIT = 1 << 4


def test_answer_needs_a_new_a_press():
    spans = np.array([[10, 30]])
    onset = np.zeros(40, np.float32)
    carried = np.zeros(40, np.int64)
    carried[5:16] = A_BIT                                # A held from before the prompt
    assert text_agreement(carried, spans) == 0.0
    out, changed = enforce_text(carried, spans, onset)
    assert text_agreement(out, spans) == 1.0 and changed.any()
    pressed = np.zeros(40, np.int64)
    pressed[20:25] = A_BIT                               # a new press inside the wait
    out, changed = enforce_text(pressed, spans, onset)
    assert np.array_equal(out, pressed) and not changed.any()


def test_text_tap_goes_where_the_head_peaks():
    spans = np.array([[10, 30]])
    onset = np.zeros(40, np.float32)
    onset[23] = 0.4
    out, changed = enforce_text(np.zeros(40, np.int64), spans, onset)
    assert np.flatnonzero(out & A_BIT).tolist() == list(range(23, 23 + TAP))
    assert np.flatnonzero(changed).tolist() == list(range(23, 23 + TAP))


def test_a_hold_through_the_whole_wait_gets_one_release():
    spans = np.array([[10, 30]])
    onset = np.zeros(40, np.float32)
    onset[20] = 0.5
    held = np.full(40, A_BIT, np.int64)                  # A never let go: no new press possible
    out, changed = enforce_text(held, spans, onset)
    assert text_agreement(out, spans) == 1.0
    assert np.flatnonzero(changed).tolist() == [19]      # released for one frame, pressed again at 20


def test_unknown_colours_are_not_dark():
    frame = _text_frame()
    frame[PROMPT_BOX] = np.where(PROMPT_GLYPH, 255, 0).astype(np.uint8)   # 255: colour not in the table
    assert prompt_visible(frame[None], COLOURS).tolist() == [False]


def test_b_also_answers_a_prompt():
    # Measured on true synthetic labels: 11 of 12 waits without an A press
    # were answered with B.
    spans = np.array([[10, 30]])
    pressed_b = np.zeros(40, np.int64)
    pressed_b[18:22] = 1 << 5
    assert text_agreement(pressed_b, spans) == 1.0


def test_a_press_just_before_the_arrow_shows_counts():
    # The game waits a frame or two before the arrow is drawn.
    spans = np.array([[10, 30]])
    early = np.zeros(40, np.int64)
    early[9:14] = A_BIT
    assert text_agreement(early, spans) == 1.0


def test_a_walk_scrolls_one_way_on_every_other_frame():
    walk = np.array([-1] * 4 + [2, -1] * 8, np.int8)
    assert step_starts(walk).tolist() == [[4, 2]]


def test_shakes_and_blips_are_not_steps():
    # Seen on the video: a hit shaking the screen up and down, and a lone
    # frame that happens to look shifted.
    shake = np.array([-1] * 4 + [1, -1, 0, -1] * 4, np.int8)
    blip = np.array([-1] * 4 + [3] + [-1] * 15, np.int8)
    assert step_starts(shake).tolist() == [] and step_starts(blip).tolist() == []


def test_a_press_on_the_first_scrolling_frame_counts():
    # The capture emulator and PyBoy can disagree by a frame.
    masks = np.zeros(20, np.int64)
    masks[10] = LEFT
    assert walking_agreement(masks, np.array([[10, 2]])) == 1.0
