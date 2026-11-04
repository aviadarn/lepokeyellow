import numpy as np

from poke.synth import BUTTONS
from scripts.label_video import TAP_FRAMES
from scripts.repair_labels import latest, runs

A, UP = 1 << BUTTONS.index("a"), 1 << BUTTONS.index("up")


def test_runs_find_every_held_stretch():
    masks = np.array([0, A, A, 0, A | UP, UP, 0, A], np.int64)
    assert runs(masks, BUTTONS.index("a")) == [(1, 3), (4, 5), (7, 8)]
    assert runs(masks, BUTTONS.index("up")) == [(4, 6)]


def test_latest_snapshot_is_before_the_edit():
    snaps = [(0, "s0"), (20, "s20"), (40, "s40")]
    assert latest(snaps, 39)[1] == "s20"
    assert latest(snaps, 40)[1] == "s40"
    assert latest(snaps, -5)[1] == "s0"


def _held(n):
    return np.full((n, 8), 0.5, np.float32)


class _FakeReplayer:
    e, offset = 0, 0

    def save(self):
        return None


def _fake_search(truth, limit):
    """A Search whose 'emulator' loses sync RUN frames after the first frame
    where the buttons differ from `truth`."""
    from scripts.repair_labels import RUN, Search

    class Fake(Search):
        def play(self, snapshot, masks, stop_at, record=False, **_):
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
    masks, lost, edits = repair(search, decoded, _held(n), onset, search.base(decoded, 0), n,
                                max_edits=10, max_backtracks=3)
    assert lost == n and np.array_equal(masks, truth)


def test_beam_finds_two_taps_closer_than_the_margin():
    from scripts.repair_labels import MARGIN, repair

    n = 1200
    truth = np.zeros(n, np.int64)
    truth[300:300 + TAP_FRAMES] = A       # two missed taps 18 frames apart:
    truth[318:318 + TAP_FRAMES] = A       # neither alone gains MARGIN frames
    assert 18 < MARGIN
    onset = np.zeros((n, 8), np.float32)
    search = _fake_search(truth, n)
    decoded = np.zeros(n, np.int64)
    masks, lost, edits = repair(search, decoded, _held(n), onset, search.base(decoded, 0), n,
                                max_edits=10, max_backtracks=3)
    assert lost == n and np.array_equal(masks, truth)


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
    from scripts.repair_labels import Search

    search = Search(_FixedLoss(), 0, 1000)
    masks = np.zeros(1000, np.int64)
    lost = search.base(masks, 0)
    assert lost == 105
    late = masks.copy()
    late[150:155] = A
    assert search.reach(late, 150, lost, 1000) == 105


def test_children_replay_from_before_the_original_loss():
    # Review focus: every replay starts at or before the first frame where
    # its buttons differ from the recorded path, however deep in the beam.
    from scripts.repair_labels import Search, beam

    search = Search(_FixedLoss(), 0, 1000)
    masks = np.zeros(1000, np.int64)
    lost = search.base(masks, 0)
    played = []
    inner = search.play

    def watched(snapshot, edited, stop_at, record=False, **rejoin):
        differ = np.flatnonzero(edited != masks)
        played.append((snapshot, int(differ[0]) if differ.size else None))
        return inner(snapshot, edited, stop_at, record, **rejoin)

    search.play = watched
    assert beam(search, masks, _held(1000), np.zeros((1000, 8), np.float32), lost, 1000,
                depth=3) is None
    assert played and all(first is None or snap <= first for snap, first in played)


def test_beam_finds_three_taps_closer_than_the_margin():
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
    masks, lost, edits = repair(search, decoded, _held(n), np.zeros((n, 8), np.float32),
                                search.base(decoded, 0), n, max_edits=10, max_backtracks=3)
    assert lost == n and np.array_equal(masks, truth)


def test_beam_turns_a_long_hold_into_a_tap_and_adds_the_missed_a():
    # 2:32 in the video: DOWN released too late, RIGHT read as a 157-frame
    # hold where the player tapped it to turn, and the A that opens the
    # girl's text missed altogether.
    from scripts.repair_labels import repair

    down, right = 1 << BUTTONS.index("down"), 1 << BUTTONS.index("right")
    n = 10_000
    truth = np.zeros(n, np.int64)
    truth[9080:9107] = down
    truth[9120:9120 + TAP_FRAMES] = right
    truth[9140:9140 + TAP_FRAMES] = A
    decoded = np.zeros(n, np.int64)
    decoded[9080:9115] = down
    decoded[9120:9277] = right
    onset = np.zeros((n, 8), np.float32)
    onset[9140, BUTTONS.index("a")] = 0.3
    search = _fake_search(truth, n)
    masks, lost, _ = repair(search, decoded, _held(n), onset, search.base(decoded, 0), n,
                            max_edits=10, max_backtracks=3)
    assert lost == n and np.array_equal(masks, truth)


def test_a_false_press_far_back_is_found_by_widening_the_window():
    # The intro's false A at 1,785 showed ~1,500 frames later.
    from scripts.repair_labels import RUN, Search, repair

    n = 2000
    decoded = np.zeros(n, np.int64)
    decoded[200:205] = A

    class Late(Search):
        def play(self, snapshot, masks, stop_at, record=False, **_):
            return min(1700 + RUN, stop_at) if (masks[200:205] & A).any() else stop_at

    search = Late(_FakeReplayer(), None, n)
    masks, lost, _ = repair(search, decoded, _held(n), np.zeros((n, 8), np.float32),
                            search.base(decoded, 0), n, max_edits=5, max_backtracks=1)
    assert lost == n and not masks.any()


def test_only_if_better_keeps_a_further_verified_reach(tmp_path):
    import h5py

    from scripts.repair_labels import should_write

    with h5py.File(tmp_path / "labels.h5", "w") as f:
        f["verified"] = np.r_[np.ones(9116, bool), np.zeros(4884, bool)]
        assert not should_write(f, 0, 5000)
        assert not should_write(f, 0, 9000)
        assert should_write(f, 0, 9500)
    with h5py.File(tmp_path / "empty.h5", "w") as f:
        assert should_write(f, 0, 10)


class _Converging:
    """An 'emulator' whose state is the last 6 frames of buttons: an edit
    the game ignores leaves no trace 6 frames after it ends. Out of sync
    from 105 whatever is pressed."""

    def __init__(self):
        self.e, self.offset, self.masks, self.steps = 0, 0, None, 0

    def save(self):
        return self.e

    def load(self, snapshot):
        self.e = snapshot

    def state_key(self):
        return self.masks[max(self.e - 6, 0):self.e].tobytes()

    def step(self):
        from scripts.sync_opening import TEXT_LIMIT

        v = self.e
        self.e += 1
        self.steps += 1
        return v, 1.0, TEXT_LIMIT if v >= 105 else 0


def test_an_edit_that_leaves_no_trace_stops_where_it_rejoins_the_path():
    from scripts.repair_labels import Search

    replayer = _Converging()
    search = Search(replayer, 0, 1000)
    masks = np.zeros(1000, np.int64)
    lost = search.base(masks, 0)
    assert lost == 105
    ignored = masks.copy()
    ignored[40:45] = A
    replayer.steps = 0
    assert search.reach(ignored, 40, lost, 700, last=44) == lost
    assert replayer.steps < 30          # rejoined at 51, not replayed to 125


def test_beam_takes_a_set_as_soon_as_it_beats_the_loss_by_the_margin():
    # Rounds past that only cost time: the repair goes on from the new loss.
    from scripts.repair_labels import beam

    n = 2000
    truth = np.zeros(n, np.int64)
    truth[300:300 + TAP_FRAMES] = A
    truth[804:804 + TAP_FRAMES] = A       # on the 6-frame grid: findable
    search = _fake_search(truth, n)
    decoded = np.zeros(n, np.int64)
    lost = search.base(decoded, 0)
    found = beam(search, decoded, _held(n), np.zeros((n, 8), np.float32), lost, n)
    assert found.what == "add a 300" and found.reach == 824


class _EnablingEdit:
    """2:32 as an 'emulator': the A that opens the girl's text works only
    once RIGHT is no longer held (the player stopped bumping into her). With
    RIGHT held the frames before 9,143 are a little off (the bump); from
    9,143 the video's text box is missing either way, unless both are fixed."""

    def __init__(self):
        self.e, self.offset, self.masks = 0, 0, None

    def save(self):
        return self.e

    def load(self, snapshot):
        self.e = snapshot

    def step(self):
        v = self.e
        self.e += 1
        m = self.masks
        bumping = bool((m[9126:9140] & (1 << BUTTONS.index("right"))).any())
        pressed = bool((m[9136:9146] & A).any()) and not (m[9130:9136] & A).any()
        talking = pressed and not bumping      # a new A press, not one held from before
        if v < 9100 or talking:
            return v, 1.0, 0
        if v < 9143:
            return v, 0.985 if bumping else 0.995, 0
        return v, 0.5, None


def test_beam_finds_a_fix_that_needs_an_edit_which_alone_gains_nothing():
    # Cutting RIGHT to a tap does not move the loss, but it brings the
    # frames before it closer to the video: the beam must keep it and add
    # the A in the next round.
    from scripts.repair_labels import MARGIN, Search, beam

    n = 10_000
    decoded = np.zeros(n, np.int64)
    decoded[9120:9277] = 1 << BUTTONS.index("right")
    onset = np.zeros((n, 8), np.float32)
    onset[9140, BUTTONS.index("a")] = 0.3
    search = Search(_EnablingEdit(), 0, n)
    lost = search.base(decoded, 0)
    assert lost == 9143
    found = beam(search, decoded, _held(n), onset, lost, n)
    assert found is not None and found.reach >= lost + MARGIN
    assert "right" in found.what and "add a" in found.what
