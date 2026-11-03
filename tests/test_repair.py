import numpy as np

from poke.synth import BUTTONS
from scripts.label_video import TAP_FRAMES
from scripts.repair_labels import far_candidates, latest, near_candidates, runs

A, UP = 1 << BUTTONS.index("a"), 1 << BUTTONS.index("up")


def test_runs_find_every_held_stretch():
    masks = np.array([0, A, A, 0, A | UP, UP, 0, A], np.int64)
    assert runs(masks, BUTTONS.index("a")) == [(1, 3), (4, 5), (7, 8)]
    assert runs(masks, BUTTONS.index("up")) == [(4, 6)]


def test_near_edits_drop_move_and_add():
    n = 400
    masks = np.zeros(n, np.int64)
    masks[300:305] = A          # a tap just before sync was lost at 320
    masks[100:110] = UP         # far before: not touched by near edits
    onset = np.zeros((n, 8), np.float32)
    onset[290, BUTTONS.index("b")] = 0.4
    found = {c.what: c for c in near_candidates(masks, onset, lost=320)}
    drop = found["drop a 300-305"]
    assert not drop.masks[300:305].any() and drop.first == 300
    moved = found["move a press 300->302"]
    assert (moved.masks[300:302] == 0).all() and (moved.masks[302:305] == A).all()
    released = found["move a release 305->309"]
    assert (released.masks[300:309] == A).all() and released.masks[309] == 0
    add_b = found["add b 290"]
    assert (add_b.masks[290:290 + TAP_FRAMES] >> BUTTONS.index("b") & 1).all()
    assert add_b.prior == np.float32(0.4)
    assert not any(c.what.startswith(("drop up", "move up")) for c in found.values())


def test_far_edits_only_drop_taps_further_back():
    n = 2000
    masks = np.zeros(n, np.int64)
    masks[200:205] = A          # the intro's kind of false tap, 1,400 frames back
    masks[1000:1100] = UP       # a walk: never dropped from far away
    onset = np.zeros((n, 8), np.float32)
    whats = [c.what for c in far_candidates(masks, onset, lost=1600)]
    assert whats == ["drop a 200-205"]


def test_latest_snapshot_is_before_the_edit():
    snaps = [(0, "s0"), (20, "s20"), (40, "s40")]
    assert latest(snaps, 39)[1] == "s20"
    assert latest(snaps, 40)[1] == "s40"
    assert latest(snaps, -5)[1] == "s0"


class _FakeReplayer:
    e, offset = 0, 0

    def save(self):
        return None


def _fake_search(truth, limit):
    """A Search whose 'emulator' loses sync RUN frames after the first frame
    where the buttons differ from `truth`."""
    from scripts.repair_labels import RUN, Search

    class Fake(Search):
        def play(self, snapshot, masks, stop_at, record=False):
            diff = np.flatnonzero(masks[:stop_at] != truth[:stop_at])
            return int(min(diff[0] + RUN, stop_at)) if len(diff) else stop_at

    return Fake(_FakeReplayer(), None, limit)


def test_repair_adds_missed_taps_and_drops_false_ones():
    from scripts.repair_labels import repair

    n = 1200
    truth = np.zeros(n, np.int64)
    truth[300:305] = A            # a tap the model missed
    truth[700:760] = A            # a hold through text
    decoded = np.zeros(n, np.int64)
    decoded[500:505] = A          # a tap the model invented
    onset = np.zeros((n, 8), np.float32)
    onset[700, BUTTONS.index("a")] = 0.3   # the head's weak hint, off the 6-frame grid
    search = _fake_search(truth, n)
    masks, lost, edits = repair(search, decoded, onset, search.base(decoded, 0), n,
                                max_edits=10, max_backtracks=3)
    assert lost == n and np.array_equal(masks, truth)
    assert [e["edit"] for e in edits] == ["add a 300", "drop a 500-505", "add a 700x60"]


LEFT, RIGHT = 1 << BUTTONS.index("left"), 1 << BUTTONS.index("right")


def test_near_edits_add_and_swap_directions():
    # The naming screen: the model read the cursor's LEFT as RIGHT, and
    # missed the small moves after it.
    n = 400
    masks = np.zeros(n, np.int64)
    masks[300:303] = RIGHT
    onset = np.zeros((n, 8), np.float32)
    found = {c.what: c for c in near_candidates(masks, onset, lost=320)}
    swap = found["swap right->left 300-303"]
    assert (swap.masks[300:303] == LEFT).all() and swap.first == 300
    up = found["add up 312"]
    assert (up.masks[312:312 + TAP_FRAMES] == UP).all() and (up.masks[300:303] == RIGHT).all()
    assert not any(w.startswith("add up") and int(w.split()[2]) < 320 - 30 for w in found)


def test_repair_chains_two_edits_when_presses_are_closer_than_the_margin():
    from scripts.repair_labels import MARGIN, repair

    n = 1200
    truth = np.zeros(n, np.int64)
    truth[300:300 + TAP_FRAMES] = A       # two missed taps 18 frames apart:
    truth[318:318 + TAP_FRAMES] = A       # neither alone gains MARGIN frames
    assert 18 < MARGIN
    onset = np.zeros((n, 8), np.float32)
    search = _fake_search(truth, n)
    decoded = np.zeros(n, np.int64)
    masks, lost, edits = repair(search, decoded, onset, search.base(decoded, 0), n,
                                max_edits=10, max_backtracks=3)
    assert lost == n and np.array_equal(masks, truth)
    assert [e["edit"] for e in edits] == ["add a 300 + add a 318"]


def test_a_rerun_clears_what_the_last_run_verified_past_its_own_reach(tmp_path):
    import h5py

    from scripts.repair_labels import write_repair

    n = 100
    path = tmp_path / "labels.h5"
    with h5py.File(path, "w") as f:
        f["buttons"] = np.zeros(n, np.uint8)
        f["buttons_ruled"] = np.zeros(n, np.uint8)
        f["buttons_repaired"] = np.full(n, A, np.uint8)
        f["verified"] = np.r_[np.ones(80, bool), np.zeros(20, bool)]   # last run: 0-80
    masks = np.zeros(n, np.int64)
    masks[:50] = UP
    with h5py.File(path, "a") as f:
        write_repair(f, "buttons_ruled", masks, first=0, lost=40, limit=90, name="strict",
                     edits=3)
    with h5py.File(path) as f:
        assert f["verified"][:40].all() and not f["verified"][40:].any()
        assert (f["buttons_repaired"][:90] == masks[:90]).all()
        assert (f["buttons_repaired"][90:] == A).all()      # outside the run: untouched
        assert "repair_strict" in f["verified"].attrs


class _FixedLoss:
    """An 'emulator' out of sync on frames 105-134 whatever is pressed: past
    the 20 frames that make it a loss, with a snapshot (120) inside them."""

    def __init__(self):
        self.e, self.offset, self.masks = 0, 0, None

    def save(self):
        return self.e

    def load(self, snapshot):
        self.e = snapshot

    def step(self):
        from scripts.sync_opening import TEXT_LIMIT

        v = self.e
        self.e += 1
        return v, 1.0, TEXT_LIMIT if 105 <= v < 135 else 0


def test_an_edit_after_the_loss_is_still_judged_from_the_loss():
    # An edit at 150 cannot fix frames 105-134; replaying it from the last
    # snapshot before 150 (120) would judge only 120-134, too short to be a
    # loss, and call it a fix.
    from scripts.repair_labels import Candidate, Search

    search = Search(_FixedLoss(), 0, 1000)
    masks = np.zeros(1000, np.int64)
    lost = search.base(masks, 0)
    assert lost == 105
    late = masks.copy()
    late[150:155] = A
    assert search.best([Candidate("add a 150", late, 150, 1.0)], lost) is None


def test_repair_chains_three_edits_when_two_are_not_enough():
    # Typing a name: a missed tap every 12 frames, so even two of them gain
    # less than MARGIN (measured on the naming screen: 12, 9, 12 frames each).
    from scripts.repair_labels import MARGIN, repair

    n = 1200
    truth = np.zeros(n, np.int64)
    for t in (300, 312, 324):
        truth[t:t + TAP_FRAMES] = A
    assert 2 * 12 < MARGIN
    search = _fake_search(truth, n)
    decoded = np.zeros(n, np.int64)
    masks, lost, edits = repair(search, decoded, np.zeros((n, 8), np.float32),
                                search.base(decoded, 0), n, max_edits=10, max_backtracks=3)
    assert lost == n and np.array_equal(masks, truth)
    assert [e["edit"] for e in edits] == ["add a 300 + add a 312 + add a 324"]
