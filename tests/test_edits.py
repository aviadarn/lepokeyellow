import numpy as np

from poke.synth import BUTTONS
from scripts.edits import LENGTHS, candidates, prior, screen
from scripts.label_video import TAP_FRAMES

KINDS = ("add", "drop", "start", "end", "length", "split", "swap")
RIGHT, DOWN, A = (BUTTONS.index(b) for b in ("right", "down", "a"))


def _probs(n, value=0.5):
    return np.full((n, 8), value, np.float32), np.zeros((n, 8), np.float32)


def _kind_and_button(what):
    kind, rest = what.split(" ", 1)
    return kind, rest.split(" ")[0].split("->")[0]


def test_every_kind_is_proposed_for_every_button():
    n = 2000
    masks = np.zeros(n, np.int64)
    for k in range(8):
        masks[1000 + 20 * k:1010 + 20 * k] |= 1 << k
    held_p, onset_p = _probs(n)
    found = {_kind_and_button(c.what) for c in candidates(masks, held_p, onset_p, 1180, 300)}
    for name in BUTTONS:
        for kind in KINDS:
            assert (kind, name) in found, (kind, name)


def test_a_long_hold_can_become_a_tap():
    n = 10_000
    masks = np.zeros(n, np.int64)
    masks[9120:9277] = 1 << RIGHT
    held_p, onset_p = _probs(n)
    found = {c.what: c for c in candidates(masks, held_p, onset_p, 9143, 300)}
    tap = found[f"length right 9120-9277->{TAP_FRAMES}"]
    held = np.flatnonzero(tap.masks >> RIGHT & 1)
    assert held.min() == 9120 and held.max() == 9120 + TAP_FRAMES - 1
    assert tap.first == 9120 + TAP_FRAMES


def test_presses_starting_before_the_window_are_editable():
    n, lost = 5000, 3000
    masks = np.zeros(n, np.int64)
    masks[lost - 400:lost - 100] = 1 << DOWN
    held_p, onset_p = _probs(n)
    ends = [c for c in candidates(masks, held_p, onset_p, lost, 300)
            if c.what.startswith("end down")]
    assert ends
    shorter = next(c for c in ends if c.what == f"end down {lost - 100}->{lost - 116}")
    assert shorter.first == lost - 116
    assert not (shorter.masks[lost - 116:lost - 100] >> DOWN & 1).any()


def test_edits_are_clipped_at_both_ends_of_the_labels():
    n = 400
    masks = np.zeros(n, np.int64)
    masks[0:3] = 1 << A
    masks[n - 2:n] = 1 << DOWN
    held_p, onset_p = _probs(n)
    for c in candidates(masks, held_p, onset_p, n - 1, 2 * n):
        edited = c.masks
        assert len(edited) == n and 0 <= c.first < n
        assert (edited >= 0).all()


def test_prior_prefers_what_the_model_believes():
    n = 400
    masks = np.zeros(n, np.int64)
    held_p, onset_p = _probs(n, 0.05)
    held_p[100:106, A] = 0.9
    found = {c.what: c for c in candidates(masks, held_p, onset_p, 120, 60)}
    assert found["add a 102"].prior > found["add b 102"].prior


def test_candidate_priors_are_the_prior_of_their_masks():
    n = 600
    rng = np.random.default_rng(1)
    masks = np.zeros(n, np.int64)
    masks[300:340] = 1 << RIGHT
    masks[350:355] = 1 << A
    held_p = rng.uniform(0, 1, (n, 8)).astype(np.float32)
    onset_p = np.zeros((n, 8), np.float32)
    for c in candidates(masks, held_p, onset_p, 360, 120)[::25]:
        assert np.isclose(c.prior, prior(masks, c.masks, held_p, 0, n), atol=1e-4), c.what


def test_screen_keeps_every_kind():
    n = 3000
    masks = np.zeros(n, np.int64)
    for k in range(8):
        masks[1500 + 30 * k:1512 + 30 * k] |= 1 << k
    held_p, onset_p = _probs(n)
    cands = candidates(masks, held_p, onset_p, 1800, 600)
    assert len(cands) >= 1000
    before = {_kind_and_button(c.what) for c in cands}
    kept = screen(cands, 48 * 2)
    assert {_kind_and_button(c.what) for c in kept} == before
    assert len(kept) <= max(96, 2 * len(before))
    assert LENGTHS[0] == TAP_FRAMES


def test_screen_keeps_edits_near_the_loss_when_the_prior_says_nothing():
    # A flat prior ranks nothing: closeness to the loss must, or a tap 20
    # frames before it is crowded out by taps 300 frames before.
    n = 1200
    masks = np.zeros(n, np.int64)
    held_p, onset_p = _probs(n)
    cands = candidates(masks, held_p, onset_p, 320, 300)
    kept = {c.what for c in screen(cands, 200, lost=320)}
    assert "add a 300" in kept and "add down 318" in kept
    assert "add a 24" not in kept
