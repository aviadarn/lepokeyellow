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
