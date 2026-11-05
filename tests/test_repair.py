import numpy as np
import pytest

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


def test_verified_reach_comes_from_a_fresh_replay():
    # The search's snapshot chain is bookkeeping: what is written as
    # verified is where a new emulator, replaying the final labels from the
    # anchor under the same rule, first loses sync.
    from scripts.repair_labels import fresh_reach

    made = []

    def new_replayer():
        made.append(_FixedLoss())
        return made[-1]

    masks = np.zeros(1000, np.int64)
    assert fresh_reach(new_replayer, masks, 0, 1000, judge_from=0) == 105
    assert len(made) == 1 and made[0].masks is masks


def test_a_stuck_point_over_its_budget_ends_the_run():
    # A wall no edit passes (the tile animation at 13,048) must stop the
    # run on its own, not search for hours.
    from scripts.repair_labels import Search, repair

    n = 3000
    plays = []

    class Wall(Search):
        def play(self, snapshot, masks, stop_at, record=False, **_):
            plays.append(1)
            self.last_cost = 0.0
            return min(320, stop_at)

    ticks = iter(range(10**6))
    search = Wall(_FakeReplayer(), None, n, clock=lambda: next(ticks))
    decoded = np.zeros(n, np.int64)
    masks, lost, edits = repair(search, decoded, _held(n), np.zeros((n, 8), np.float32),
                                search.base(decoded, 0), n, max_edits=5, max_backtracks=3,
                                stuck_budget=100)
    assert lost == 320 and not edits and len(plays) < 150


class _HiddenWall:
    """An 'emulator' with one hidden byte: out of sync from 300 unless 0xFFD8
    holds 7 by then (the tile-animation wall at 13,048), and, with `second`,
    out of sync from 500 whatever is done."""

    def __init__(self, second=False):
        self.e, self.offset, self.masks, self.writes = 0, 0, None, {}
        self.memory, self.second = {0xFFD8: 0, 0xD084: 0}, second

    def save(self):
        return self.e, dict(self.memory)

    def load(self, snapshot):
        self.e, memory = snapshot
        self.memory = dict(memory)

    def step(self):
        from scripts.sync_opening import TEXT_LIMIT

        v = self.e
        for address, value in self.writes.get(v, ()):
            self.memory[address] = value
        self.e += 1
        stuck = (v >= 300 and self.memory[0xFFD8] != 7) or (self.second and v >= 500)
        return v, 1.0, TEXT_LIMIT if stuck else 0


def test_resync_finds_the_write_that_passes_a_wall():
    from scripts.repair_labels import TILE_ANIMATION, Search, repair

    n = 1000
    search = Search(_HiddenWall(), (0, {0xFFD8: 0, 0xD084: 0}), n)
    masks = np.zeros(n, np.int64)
    lost = search.base(masks, 0)
    assert lost == 300
    out, lost, edits = repair(search, masks, _held(n), np.zeros((n, 8), np.float32), lost, n,
                              max_edits=5, max_backtracks=2, max_window=300,
                              tiers=(TILE_ANIMATION,))
    assert lost == n and edits[-1]["edit"].startswith("write FFD8=7")
    assert any((0xFFD8, 7) in pairs for pairs in search.best_writes.values())


def test_backtracking_removes_a_committed_write():
    from scripts.repair_labels import TILE_ANIMATION, Search, repair

    n = 1000
    search = Search(_HiddenWall(second=True), (0, {0xFFD8: 0, 0xD084: 0}), n)
    masks = np.zeros(n, np.int64)
    lost = search.base(masks, 0)
    _, lost, _ = repair(search, masks, _held(n), np.zeros((n, 8), np.float32), lost, n,
                        max_edits=5, max_backtracks=1, max_window=300,
                        tiers=(TILE_ANIMATION,))
    assert lost == 500                                   # the write's reach is kept as best
    assert search.writes == {}                           # ... but undone on the stack
    assert any((0xFFD8, 7) in pairs for pairs in search.best_writes.values())


def _labels_file(path, n=1000):
    import h5py

    with h5py.File(path, "w") as f:
        f["buttons"] = np.zeros(n, np.uint8)
        f["buttons_ruled"] = np.zeros(n, np.uint8)
        f["buttons_repaired"] = np.zeros(n, np.uint8)
        f["verified"] = np.zeros(n, bool)
        f["verified_resync"] = np.r_[np.zeros(200, bool), np.ones(100, bool), np.zeros(n - 300, bool)]
        f["state_writes"] = np.array([[250, 0xFFD8, 3], [950, 0xD084, 1]], np.int32)


def test_write_repair_splits_verified_at_the_first_write(tmp_path):
    import h5py

    from scripts.repair_labels import write_repair

    _labels_file(tmp_path / "l.h5")
    with h5py.File(tmp_path / "l.h5", "a") as f:
        write_repair(f, "buttons_ruled", np.zeros(1000, np.int64), 0, 800, 900, "r", 2,
                     writes={500: ((0xFFD8, 7),)})
        assert f["verified"][:500].all() and not f["verified"][500:].any()
        assert f["verified_resync"][500:800].all() and not f["verified_resync"][:500].any()
        assert not f["verified_resync"][800:].any()
        # from the run's start on, only its own write is left (the chain changed)
        assert f["state_writes"][:].tolist() == [[500, 0xFFD8, 7]]


def test_a_run_without_writes_clears_old_writes_in_its_stretch(tmp_path):
    import h5py

    from scripts.repair_labels import write_repair

    _labels_file(tmp_path / "l.h5")
    with h5py.File(tmp_path / "l.h5", "a") as f:
        write_repair(f, "buttons_ruled", np.zeros(1000, np.int64), 0, 600, 900, "r", 0)
        assert f["verified"][:600].all() and not f["verified_resync"][:].any()
        assert f["state_writes"][:].tolist() == []


def test_fresh_reach_applies_the_writes():
    from scripts.repair_labels import fresh_reach

    masks = np.zeros(1000, np.int64)
    assert fresh_reach(_HiddenWall, masks, 0, 1000, judge_from=0) == 300
    assert fresh_reach(_HiddenWall, masks, 0, 1000, judge_from=0,
                       writes={260: ((0xFFD8, 7),)}) == 1000


def test_a_cheap_resync_comes_before_widening_the_window():
    # At 13,048 widening the window to 2,400 frames used the whole budget;
    # the tile tier (168 values) must be tried first.
    import itertools

    from scripts.repair_labels import TILE_ANIMATION, Search, repair

    n = 1000
    ticks = itertools.count()
    search = Search(_HiddenWall(), (0, {0xFFD8: 0, 0xD084: 0}), n, clock=lambda: next(ticks))
    masks = np.zeros(n, np.int64)
    lost = search.base(masks, 0)
    _, lost, edits = repair(search, masks, _held(n), np.zeros((n, 8), np.float32), lost, n,
                            max_edits=5, max_backtracks=1, max_window=2400,
                            stuck_budget=3000, tiers=(TILE_ANIMATION,))
    assert lost == n and edits[0]["edit"].startswith("write FFD8=7")


def test_the_narrow_beam_leaves_the_resyncs_half_the_budget():
    # Window 1's battle (Mt. Moon): the narrow beam used 860 of the wall's
    # 900 CPU-s; the resyncs after it ran out, and the random tier never ran.
    # Here the narrow beam takes about 2,000 ticks, and the tile tier reaches
    # FFD8=7 at its 57th value.
    import itertools

    from scripts.repair_labels import TILE_ANIMATION, Search, repair

    n = 1000
    ticks = itertools.count()
    search = Search(_HiddenWall(), (0, {0xFFD8: 0, 0xD084: 0}), n, clock=lambda: next(ticks))
    masks = np.zeros(n, np.int64)
    lost = search.base(masks, 0)
    _, lost, edits = repair(search, masks, _held(n), np.zeros((n, 8), np.float32), lost, n,
                            max_edits=5, max_backtracks=1, max_window=300,
                            stuck_budget=2040, tiers=(TILE_ANIMATION,))
    assert lost == n and edits[0]["edit"].startswith("write FFD8=7")


def test_a_costly_resync_comes_before_widening_the_window(monkeypatch):
    # Window 1 (Mt. Moon): widening to 600 and 1,200 frames used the whole
    # budget, so the random tier (a minute's sweep on workers) never ran.
    import scripts.repair_labels as rl

    calls = []
    narrow = rl.beam

    def beam(search, masks, held_p, onset_p, lost, limit, width, depth, window, *rest, **kw):
        calls.append(("beam", window))
        return narrow(search, masks, held_p, onset_p, lost, limit, width, depth, window, *rest, **kw)

    def sweep(replayer, snapshot, cap, registers, workers, writes, at, judge_from, run,
              deadline=None):
        calls.append(("sweep",))
        return [(cap, (7, 0))]

    monkeypatch.setattr(rl, "beam", beam)
    n = 1000
    search = rl.Search(_HiddenWall(), (0, {0xFFD8: 0, 0xD084: 0, 0xFFD4: 0}), n)
    masks = np.zeros(n, np.int64)
    lost = search.base(masks, 0)
    big = ((0xFFD8, range(40)), (0xFFD4, range(40)))      # 1,600 values: a worker tier
    _, lost, edits = rl.repair(search, masks, _held(n), np.zeros((n, 8), np.float32), lost, n,
                               max_edits=5, max_backtracks=1, max_window=2400,
                               tiers=(big,), sweep=sweep)
    assert lost == n and edits[0]["edit"].startswith("write FFD8=7")
    assert calls[:2] == [("beam", 300), ("sweep",)]
    assert ("beam", 600) not in calls


def test_only_if_better_counts_resync_verified_frames(tmp_path):
    import h5py

    from scripts.repair_labels import should_write

    with h5py.File(tmp_path / "l.h5", "w") as f:
        f["verified"] = np.r_[np.ones(13008, bool), np.zeros(10000, bool)]
        f["verified_resync"] = np.r_[np.zeros(13008, bool), np.ones(7273, bool),
                                     np.zeros(23008 - 20281, bool)]
        assert not should_write(f, 0, 13048)       # a buttons-only 13,048 is not better
        assert should_write(f, 0, 21000)


def test_a_run_clears_every_claim_after_its_start(tmp_path):
    # The replay is chained: once a run changes buttons from `first`, nothing
    # after it stays verified, whatever an older run reached.
    import h5py

    from scripts.repair_labels import write_repair

    _labels_file(tmp_path / "l.h5", n=1000)
    with h5py.File(tmp_path / "l.h5", "a") as f:
        f["verified_resync"][:] = np.r_[np.zeros(700, bool), np.ones(300, bool)]
        write_repair(f, "buttons_ruled", np.zeros(1000, np.int64), 0, 400, 600, "r", 0)
        assert not f["verified_resync"][400:].any() and not f["verified"][400:].any()
        assert f["state_writes"][:].tolist() == []


def test_resync_searches_the_current_labels_within_its_budget():
    # The large tier runs on workers: it must be given the labels being
    # repaired (not the last candidate played), and not start when the
    # stuck point is out of time.
    from scripts.repair_labels import Search, resync

    n = 1000
    seen = []

    def sweep(replayer, snapshot, cap, registers, workers, writes, at, judge_from, run,
              deadline=None):
        seen.append(replayer.masks.copy())
        return []

    start = (0, {0xFFD8: 0, 0xD084: 0, 0xFFD3: 0, 0xFFD4: 0})
    search = Search(_HiddenWall(), start, n)
    masks = np.zeros(n, np.int64)
    lost = search.base(masks, 0)
    other = masks.copy()
    other[50:55] = A
    search.reach(other, 50, lost, n)                  # leaves another candidate's labels
    big = ((0xFFD3, range(40)), (0xFFD4, range(40)))  # 1,600 values: a worker tier
    resync(search, masks, lost, n, (big,), sweep=sweep)
    assert seen and all(np.array_equal(m, masks) for m in seen)
    calls = len(seen)
    search.deadline, search.clock = 0.0, lambda: 1.0  # out of time
    resync(search, masks, lost, n, (big,), sweep=sweep)
    assert len(seen) == calls


def _register_sweep(reach_of, sizes):
    """A sweep that scores each combination with `reach_of` and records how
    many combinations each call was given."""
    import itertools

    def sweep(replayer, snapshot, cap, registers, workers, writes, at, judge_from, run,
              deadline=None):
        combos = list(itertools.product(*[list(values) for _, values in registers]))
        sizes.append(len(combos))
        return sorted(((min(reach_of(c), cap), c) for c in combos), key=lambda r: (-r[0], r[1]))
    return sweep


def test_a_large_tier_is_searched_one_register_at_a_time():
    # Window 1 (Mt. Moon): hRandomAdd decides the encounter, then hRandomSub
    # the species: 256 + 256 replays found what the 65,536 took 8 minutes for.
    from scripts.repair_labels import RANDOM, Search, resync

    n, sizes = 1000, []
    search = Search(_HiddenWall(), (0, {0xFFD8: 0, 0xD084: 0, 0xFFD3: 117, 0xFFD4: 111}), n)
    masks = np.zeros(n, np.int64)
    lost = search.base(masks, 0)
    found = resync(search, masks, lost, n, (RANDOM,), sweep=_register_sweep(
        lambda c: lost + 600 if c == (119, 97) else lost + 140 if c[0] == 119 else lost, sizes))
    assert found.what.startswith("write FFD3=119 FFD4=97")
    assert sizes == [256, 256]


def test_a_large_tier_falls_back_to_every_combination():
    # Neither register alone plays further: only the pair does.
    from scripts.repair_labels import RANDOM, Search, resync

    n, sizes = 1000, []
    search = Search(_HiddenWall(), (0, {0xFFD8: 0, 0xD084: 0, 0xFFD3: 117, 0xFFD4: 111}), n)
    masks = np.zeros(n, np.int64)
    lost = search.base(masks, 0)
    found = resync(search, masks, lost, n, (RANDOM,), sweep=_register_sweep(
        lambda c: lost + 600 if c == (3, 5) else lost, sizes))
    assert found.what.startswith("write FFD3=3 FFD4=5")
    assert sizes == [256, 256, 65536]


def test_a_one_register_large_tier_is_swept_once():
    from scripts.repair_labels import Search, resync

    n, sizes = 1000, []
    search = Search(_HiddenWall(), (0, {0xFFD8: 0, 0xD084: 0, 0xC000: 0}), n)
    masks = np.zeros(n, np.int64)
    lost = search.base(masks, 0)
    assert resync(search, masks, lost, n, (((0xC000, range(2000)),),),
                  sweep=_register_sweep(lambda c: lost, sizes)) is None
    assert sizes == [2000]


def test_a_wall_keeps_its_budget_through_backtracking():
    # Window 3: alternative START taps all reach the same wall; each one
    # used to start the wall's budget afresh.
    import itertools

    from scripts.repair_labels import Search, repair

    n = 2000
    ticks = itertools.count()

    class SameWall(Search):
        def play(self, snapshot, masks, stop_at, record=False, **_):
            # Each tap reaches the wall a frame or two apart: an undone tap's
            # reach never quite equals its alternative's.
            self.last_cost = 0.0
            taps = np.flatnonzero(masks[290:312] & A)
            return min(500 + int(taps[0]) % 5 if len(taps) else 320, stop_at)

    search = SameWall(_FakeReplayer(), None, n, clock=lambda: next(ticks))
    decoded = np.zeros(n, np.int64)
    repair(search, decoded, _held(n), np.zeros((n, 8), np.float32), search.base(decoded, 0), n,
           max_edits=20, max_backtracks=8, max_window=300, stuck_budget=3000)
    # One stuck search here is one narrow beam (no widening): the budget
    # outlives it, so the backtracking runs -- and must still stop within
    # one budget for the wall, not one per alternative that reaches it.
    assert next(ticks) < 1.5 * 3000


def test_resync_tiers_parse_from_their_names():
    import pytest

    from scripts.repair_labels import MENU_CURSOR, RANDOM, TILE_ANIMATION, parse_tiers

    assert parse_tiers("tile,menu") == (TILE_ANIMATION, MENU_CURSOR)
    assert parse_tiers("random") == (RANDOM,)
    assert parse_tiers("badges") == (((0xD355, range(256)),),)        # wObtainedBadges
    assert parse_tiers("none") == ()
    with pytest.raises(ValueError):
        parse_tiers("tiles")


ROM_PATH, VIDEO_PATH = __import__("pathlib").Path("pokeyellow.gbc"), __import__("pathlib").Path("data/longplay/video.h5")


@pytest.mark.skipif(not (ROM_PATH.exists() and VIDEO_PATH.exists()), reason="needs the ROM and the video")
def test_a_windows_screens_rebuild_its_party_and_bag():
    import h5py

    from poke.env import PokeEnv
    from poke.symbols import load
    from scripts.repair_labels import rebuild_from_screens

    env = PokeEnv(str(ROM_PATH), "states/cgb/teleport/CeladonCity_w6.state", color=True)
    env.reset()
    with h5py.File(VIDEO_PATH) as h5:
        rebuilt = rebuild_from_screens(env.pyboy, h5, 659260, 1800, ROM_PATH.read_bytes(), load())
    assert [p["level"] for p in rebuilt["party"]] == [26, 27, 28, 26, 26, 25]
    assert 659860 <= rebuilt["party_frame"] <= 659960
    assert env.pyboy.memory[0xD162] == 6
    # Ivysaur's submenu (3:03:30) lists CUT: it knows it.
    assert rebuilt["field_moves"] == {5: ["CUT"]}
    cut = 0x0F
    assert cut in [env.pyboy.memory[0xD16A + 5 * 44 + 8 + i] for i in range(4)]


@pytest.mark.skipif(not (ROM_PATH.exists() and VIDEO_PATH.exists()), reason="needs the ROM and the video")
def test_field_moves_go_to_the_member_named_not_the_slot_after_a_switch():
    # Celadon: the player later switches Ivysaur and Pidgeotto, whose
    # submenus then show in slots 1 and 5. The moves belong to the members
    # as the first party read has them.
    import h5py

    from poke.env import PokeEnv
    from poke.symbols import load
    from scripts.repair_labels import rebuild_from_screens

    env = PokeEnv(str(ROM_PATH), "states/cgb/teleport/CeladonCity_w6.state", color=True)
    env.reset()
    with h5py.File(VIDEO_PATH) as h5:
        rebuilt = rebuild_from_screens(env.pyboy, h5, 659260, 8400, ROM_PATH.read_bytes(), load())
    assert rebuilt["party"][1]["name"] == "PIDGEOTTO" and rebuilt["party"][5]["name"] == "IVYSAUR"
    assert rebuilt["field_moves"] == {5: ["CUT"], 1: ["FLY"]}


@pytest.mark.skipif(not (ROM_PATH.exists() and VIDEO_PATH.exists()), reason="needs the ROM and the video")
def test_a_battles_move_menu_rebuilds_that_members_moves():
    # Mt. Moon, 0:43: Nidoran♀'s FIGHT menu lists SCRATCH, TACKLE, GROWL and
    # DOUBLE KICK; the rebuild had given it its starting moves.
    import h5py

    from poke.env import PokeEnv
    from poke.party import moves_by_name
    from poke.symbols import load

    from scripts.repair_labels import rebuild_from_screens

    rom, symbols = ROM_PATH.read_bytes(), load()
    env = PokeEnv(str(ROM_PATH), "states/cgb/teleport/MtMoon1F_w0.state", color=True)
    env.reset()
    with h5py.File(VIDEO_PATH) as h5:
        rebuilt = rebuild_from_screens(env.pyboy, h5, 152796, 2100, rom, symbols)
    assert rebuilt["party"][2]["name"] == "NIDORAN♀"
    moves = ["SCRATCH", "TACKLE", "GROWL", "DOUBLE KICK"]
    assert rebuilt["moves"] == {2: moves}
    assert rebuilt["pp"][2][0] == 35 and rebuilt["pp"][2][2] == 40
    ids, at = moves_by_name(rom, symbols), 0xD16A + 2 * 44
    assert [env.pyboy.memory[at + 8 + i] for i in range(4)] == [ids[n] for n in moves]
    # ... and the same again from a report's JSON, as a fresh replay or a viewer applies it.
    import json

    from scripts.repair_labels import apply_rebuilt

    fresh = PokeEnv(str(ROM_PATH), "states/cgb/teleport/MtMoon1F_w0.state", color=True)
    fresh.reset()
    apply_rebuilt(fresh.pyboy, json.loads(json.dumps(rebuilt)), rom, symbols)
    party = range(0xD162, 0xD2F7)
    assert [fresh.pyboy.memory[a] for a in party] == [env.pyboy.memory[a] for a in party]


@pytest.mark.skipif(not (ROM_PATH.exists() and VIDEO_PATH.exists()), reason="needs the ROM and the video")
def test_a_bag_opened_over_the_start_menu_is_rebuilt_with_its_quantities():
    # Viridian, 6:05:48: EXIT of the START menu shows below the bag's list.
    import h5py

    from poke.env import PokeEnv
    from poke.symbols import load
    from scripts.repair_labels import rebuild_from_screens

    env = PokeEnv(str(ROM_PATH), "states/cgb/teleport/ViridianCity_w0.state", color=True)
    env.reset()
    with h5py.File(VIDEO_PATH) as h5:
        rebuilt = rebuild_from_screens(env.pyboy, h5, 1315481, 200, ROM_PATH.read_bytes(), load())
    assert [tuple(item) for item in rebuilt["bag"]] == [
        ("BICYCLE", 1), ("SUPER POTION", 1), ("GREAT BALL", 6), ("REVIVE", 11)]
    m = env.pyboy.memory
    assert [m[0xD31D + 2 * i + 1] for i in range(4)] == [1, 1, 6, 11]


@pytest.mark.skipif(not (ROM_PATH.exists() and VIDEO_PATH.exists()), reason="needs the ROM and the video")
def test_the_start_menu_is_not_taken_for_the_bag():
    # Mt. Moon, 0:42:33: the START menu's OPTION read as a one-item list.
    import h5py

    from poke.env import PokeEnv
    from poke.symbols import load
    from scripts.repair_labels import rebuild_from_screens

    env = PokeEnv(str(ROM_PATH), "states/cgb/teleport/MtMoon1F_w0.state", color=True)
    env.reset()
    with h5py.File(VIDEO_PATH) as h5:
        rebuilt = rebuild_from_screens(env.pyboy, h5, 153040, 40, ROM_PATH.read_bytes(), load())
    assert "bag" not in rebuilt


@pytest.mark.skipif(not (ROM_PATH.exists() and VIDEO_PATH.exists()), reason="needs the ROM and the video")
def test_a_rebuild_can_be_applied_again_to_a_fresh_emulator():
    # The fresh replay that confirms a window starts from a new emulator: it
    # must carry the same rebuilt party, or it stops at the party screen.
    import h5py

    from poke.env import PokeEnv
    from poke.symbols import load
    from scripts.repair_labels import apply_rebuilt, rebuild_from_screens

    rom, symbols = ROM_PATH.read_bytes(), load()
    first = PokeEnv(str(ROM_PATH), "states/cgb/teleport/MtMoon1F_w0.state", color=True)
    first.reset()
    with h5py.File(VIDEO_PATH) as h5:
        rebuilt = rebuild_from_screens(first.pyboy, h5, 152796, 600, rom, symbols)
    fresh = PokeEnv(str(ROM_PATH), "states/cgb/teleport/MtMoon1F_w0.state", color=True)
    fresh.reset()
    apply_rebuilt(fresh.pyboy, rebuilt, rom, symbols)
    party = range(0xD162, 0xD2F7)               # count, species, structs, OT names, nicknames
    assert [fresh.pyboy.memory[a] for a in party] == [first.pyboy.memory[a] for a in party]


@pytest.mark.skipif(not (ROM_PATH.exists() and VIDEO_PATH.exists()), reason="needs the ROM and the video")
def test_a_rebuild_read_back_from_its_report_applies_its_field_moves():
    # A report stores the rebuild as JSON: the slots of its field moves come back as text.
    import json

    import h5py

    from poke.env import PokeEnv
    from poke.symbols import load
    from scripts.repair_labels import apply_rebuilt, rebuild_from_screens

    rom, symbols = ROM_PATH.read_bytes(), load()
    first = PokeEnv(str(ROM_PATH), "states/cgb/teleport/CeladonCity_w6.state", color=True)
    first.reset()
    with h5py.File(VIDEO_PATH) as h5:
        rebuilt = rebuild_from_screens(first.pyboy, h5, 659860, 120, rom, symbols)
    assert rebuilt["field_moves"] == {5: ["CUT"]}
    fresh = PokeEnv(str(ROM_PATH), "states/cgb/teleport/CeladonCity_w6.state", color=True)
    fresh.reset()
    apply_rebuilt(fresh.pyboy, json.loads(json.dumps(rebuilt)), rom, symbols)
    party = range(0xD162, 0xD2F7)
    assert [fresh.pyboy.memory[a] for a in party] == [first.pyboy.memory[a] for a in party]
