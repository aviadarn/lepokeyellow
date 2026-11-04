"""Repair the labels where replaying them stops matching the video.

The IDM's buttons are a guess; the emulator is the judge. Replay the labels
from a state that matches the video (power-on, or a save state such as the
bedroom), and at the first place they stop matching, try small edits to the
buttons -- drop a press, add an A / B / START tap, move a press or its release
by a few frames -- and keep the edit that plays in sync longest. Then go on
from there.

What comes out is buttons the emulator has checked frame by frame, up to the
first place no small edit explains: a random event the two emulators roll
differently, an edit cut, or a mistake further back than the search looks.

Search, at each loss of sync (first of RUN frames in a row out of sync: under
SYNC_SHARE, or text that differs, scripts/sync_opening.in_sync): a beam over
sets of edits from one grammar for every button (scripts/edits.py -- add,
drop, move a start or end by up to 256 frames, set a length, split, swap
for another button), tried in the order the IDM finds most plausible. Each
member of the beam expands at its own loss of sync, so small gains add up:
typing a name is a tap every ~12 frames, and each one alone buys under
MARGIN. A set is taken when it plays HORIZON frames past the old loss, or,
after the rounds stop gaining, the furthest one if it beat the old loss by
MARGIN. Where nothing helps, the window doubles (a wrong tap can sit unseen:
the intro's false A at 1,785 shows ~1,500 frames on), then backtracking
undoes recent sets. Nothing in it knows about walking, menus or text.

Writes `buttons_repaired` and `verified` into the labels file, and the edits
to results/repair_<name>.json.

  uv run python -m scripts.repair_labels --frames 20000
  uv run python -m scripts.repair_labels --anchor-state states/cgb/bedroom.state --name bedroom
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import h5py
import numpy as np

from poke.synth import BUTTONS
from poke.video import Snapper
from scripts.edits import Candidate, _peaks, _set, candidates, runs, screen  # noqa: F401
from scripts.sync_opening import (Replayer, VideoWindow, align, anchored, boot, find_anchor,
                                  in_sync, release_carried, rows_from_writes,
                                  writes_from_rows)

RUN = 20
HORIZON = 600
MARGIN = 30
SNAPSHOT_EVERY = 20
REJOIN = 120        # frames after an edit to look for the recorded path again
REJOIN_EVERY = 4    # ... every this many frames: a state digest costs 0.2 ms
COST_SPAN = 40      # frames before a loss over which sets that lose there are compared
COST_EPS = 0.005    # a closer match than this counts as getting nearer the video
NEUTRAL = 2         # rounds a beam may go on getting nearer without getting further
# Hidden state a button cannot reach, tried in this order where no edit helps:
TILE_ANIMATION = ((0xFFD8, range(21)), (0xD084, range(8)))   # flowers and water (13,048)
RANDOM = ((0xFFD3, range(256)), (0xFFD4, range(256)))         # hRandomAdd, hRandomSub
RESYNC_LEAD = 40    # frames before the loss a write goes in
IN_PROCESS = 1000   # tiers this small are searched here, larger ones on workers


def latest(snapshots: list[tuple[int, tuple]], frame: int):
    """The last snapshot taken before video frame `frame` was pressed."""
    usable = [s for s in snapshots if s[0] <= frame]
    return usable[-1] if usable else snapshots[0]


class Search:
    def __init__(self, replayer: Replayer, start: tuple, limit: int, run: int = RUN,
                 judge_from: int = 0, clock=time.process_time):
        self.rep, self.limit, self.run = replayer, limit, run
        # CPU time, and the time a stuck point's search must end by (None: no limit).
        self.clock, self.deadline = clock, None
        # State writes committed so far, and those of the furthest labels.
        self.writes: dict[int, tuple] = {}
        self.best_writes: dict[int, tuple] = {}
        # Frames before this are played but not judged: from power-on, the
        # boot screens of PyBoy and the capture emulator differ for ~50
        # frames whatever is pressed.
        self.judge_from = judge_from
        self.snapshots = [(self._next(), start)]
        self.played = 0
        # The recorded path's state after each frame (`Replayer.state_key`):
        # a candidate whose state equals it again will play exactly as it
        # did, so its reach is the path's.
        self.keys: dict[int, bytes] = {}
        # ... and how far each of its frames was from the video (1 - share).
        self.base_cost: dict[int, float] = {}
        self.last_cost = 0.0

    def _next(self) -> int:
        return self.rep.e + self.rep.offset

    def play(self, snapshot: tuple, masks: np.ndarray, stop_at: int, record: bool = False,
             rejoin_after: int | None = None, rejoin_reach: int | None = None,
             cost_from: int | None = None, writes: dict | None = None) -> int:
        """Video frame where sync is lost (first of `run` frames out of
        sync), or `stop_at`. With `rejoin_after`, once past that frame and
        before `rejoin_reach`, a state equal to the recorded path's returns
        `rejoin_reach` at once (checked for REJOIN frames). With `cost_from`,
        `last_cost` is how far the frames from there on were from the video,
        the recorded path's figures standing in for frames not replayed."""
        self.rep.load(snapshot)
        self.rep.masks = masks
        self.rep.writes = self.writes if writes is None else writes
        keyed = hasattr(self.rep, "state_key")
        bad, first_bad = 0, None
        cost, started = 0.0, False

        def done(v_end: int, rejoined_at: int | None = None) -> int:
            total = cost
            if rejoined_at is not None:
                total += sum(c for u, c in self.base_cost.items() if u > rejoined_at)
            self.last_cost = total
            return v_end

        while True:
            if record and self._next() % SNAPSHOT_EVERY == 0 and self._next() > self.snapshots[-1][0]:
                self.snapshots.append((self._next(), self.rep.save()))
            out = self.rep.step()
            self.played += 1
            if out is None:
                if self._next() >= stop_at:
                    return stop_at
                continue
            v, share, text = out
            if v >= stop_at:
                return done(stop_at)
            if cost_from is not None and v >= cost_from:
                if not started and v > cost_from:
                    cost += sum(self.base_cost.get(u, 0.0) for u in range(cost_from, v))
                started = True
                cost += 1.0 - share
            if record:
                self.base_cost[v] = 1.0 - share
                if keyed:
                    self.keys[v] = self.rep.state_key()
            elif (rejoin_after is not None and keyed and rejoin_after < v < rejoin_reach
                  and v - rejoin_after <= REJOIN and (v - rejoin_after) % REJOIN_EVERY == 0
                  and self.keys.get(v) == self.rep.state_key()):
                return done(rejoin_reach, v)
            if not in_sync(share, text) and v >= self.judge_from:
                bad += 1
                first_bad = v if first_bad is None else first_bad
                if bad >= self.run:
                    return done(first_bad)
            else:
                bad, first_bad = 0, None

    def base(self, masks: np.ndarray, frame: int) -> int:
        """Play the current labels from the last snapshot before `frame`,
        keeping snapshots on the way; drop the ones after it first."""
        self.snapshots = [s for s in self.snapshots if s[0] <= frame] or self.snapshots[:1]
        start = self.snapshots[-1][0]
        self.keys = {v: k for v, k in self.keys.items() if v < start}
        self.base_cost = {v: c for v, c in self.base_cost.items() if v < start}
        return self.play(self.snapshots[-1][1], masks, self.limit, record=True)

    def out_of_time(self) -> bool:
        return self.deadline is not None and self.clock() > self.deadline

    def reach(self, masks: np.ndarray, first: int, lost: int, cap: int,
              last: int | None = None, writes: dict | None = None) -> int:
        """Where `masks` lose sync, played from the last snapshot at or before
        both its first edited frame and `lost`: starting inside the lost
        frames would skip them, and call an edit that cannot touch them (a
        later release) a fix. Given `last`, its last edited frame, an edit
        the game shrugs off stops where it rejoins the recorded path."""
        return self.play(latest(self.snapshots, min(first, lost))[1], masks, cap,
                         rejoin_after=last, rejoin_reach=lost, cost_from=lost - COST_SPAN,
                         writes=writes)


def beam(search: Search, masks: np.ndarray, held_p: np.ndarray, onset_p: np.ndarray,
         lost: int, limit: int, width: int = 4, depth: int = 12, window: int = 300,
         cap: int = 400, exclude: frozenset = frozenset(), avoid: int | None = None):
    """The set of edits that plays furthest past `lost`, or None.

    Each round expands every member at its own loss of sync with the best
    `cap` edits (`screen`), and keeps the `width` that reach furthest (then
    the likelier, then the shorter). A set reaching HORIZON frames past
    `lost` is taken at once, and so is a round's best once it beats `lost`
    by MARGIN; rounds go on only while the gain is smaller. Sets that lose
    sync at the same frame are ranked by how near the video their frames
    stay before it, and a round that gets no further but nearer may go on
    (NEUTRAL times): some fixes need an edit that alone gains nothing. `exclude` names
    sets already tried, `avoid` a reach to skip (an undone set's: in effect
    the same set)."""
    stop = min(limit, lost + HORIZON)
    root = Candidate("", masks=masks, first=lost, prior=0.0, reach=lost)
    root.cost = sum(c for u, c in search.base_cost.items() if u >= lost - COST_SPAN)
    frontier = [root]
    best, seen = None, set()
    nearest, neutral = root.cost, 0
    for _ in range(depth):
        children = []
        for member in frontier:
            base = member.masks
            for c in screen(candidates(base, held_p, onset_p, member.reach, window), cap,
                            lost=member.reach):
                if search.out_of_time():
                    break
                what = f"{member.what} + {c.what}" if member.what else c.what
                if what in exclude:
                    continue
                edited = c.masks
                key = hash(edited.tobytes())
                if key in seen:
                    continue
                seen.add(key)
                first = min(member.first, c.first)
                child = Candidate(what, first=first, prior=member.prior + c.prior,
                                  base=base, ops=c.ops)
                differ = np.flatnonzero(edited != masks)
                last = int(differ[-1]) if differ.size else None
                child.reach = search.reach(edited, first, lost, stop, last)
                child.cost = search.last_cost
                if child.reach == avoid:
                    continue
                if child.reach >= stop:
                    return child
                children.append(child)
        if not children or search.out_of_time():
            break
        children.sort(key=lambda c: (-c.reach, c.cost, -c.prior, c.what.count(" + ")))
        top = children[0]
        if best is None or top.reach > best.reach:
            best, neutral = top, 0
        elif top.reach == best.reach and top.cost < nearest - COST_EPS and neutral < NEUTRAL:
            # No further, but nearer the video: an edit that only enables the
            # next one (a RIGHT held into the girl must stop before her A
            # can open the text).
            neutral += 1
        else:
            break
        nearest = min(nearest, top.cost)
        frontier = children[:width]
        if best.reach >= lost + MARGIN:
            # Enough to take: more rounds only cost time, and the repair
            # goes on from the new loss of sync anyway.
            return best
    return best if best is not None and best.reach >= lost + MARGIN else None


def _with(writes: dict, frame: int, pairs) -> dict:
    out = dict(writes)
    out[frame] = out.get(frame, ()) + tuple(pairs)
    return out


def resync(search: Search, masks: np.ndarray, lost: int, limit: int, tiers,
           exclude: frozenset = frozenset(), avoid: int | None = None, workers: int = 4,
           sweep=None):
    """A write of hidden state, RESYNC_LEAD frames before `lost`, that plays
    MARGIN frames past it, or None: for each tier ((address, values), ...)
    in turn, every combination is tried, the furthest-reaching kept (ties:
    the smallest change from the values the emulator has there)."""
    at = max(lost - RESYNC_LEAD, search.judge_from, 0)
    snapshot = latest(search.snapshots, at)
    cap = min(search.limit, lost + HORIZON)
    search.rep.load(snapshot[1])
    memory = getattr(search.rep, "pyboy", search.rep).memory
    for tier in tiers:
        addresses = [a for a, _ in tier]
        now = [int(memory[a]) for a in addresses]
        combos = list(itertools.product(*[list(values) for _, values in tier]))
        if len(combos) <= IN_PROCESS:
            ranked = []
            for combo in combos:
                if search.out_of_time():
                    break
                ranked.append((search.reach(masks, at, lost, cap, last=at,
                                            writes=_with(search.writes, at, zip(addresses, combo))),
                               combo))
        else:
            if search.out_of_time():
                return None
            if sweep is None:
                from scripts.rng_sync import find_state as sweep
            # The workers read the replayer's labels: the ones being repaired,
            # not the last candidate played. Their CPU is not this process's,
            # so the stuck budget left is turned into a wall-clock deadline.
            search.rep.masks = masks
            deadline = None
            if search.deadline is not None:
                deadline = time.time() + max(search.deadline - search.clock(), 0) / workers
            ranked = sweep(search.rep, snapshot[1], cap, tier, workers, search.writes, at,
                           search.judge_from, search.run, deadline=deadline)
            if deadline is not None and time.time() > deadline:
                search.deadline = search.clock() - 1      # the sweep spent the budget
        best = None
        for reach, combo in ranked:
            what = "write " + " ".join(f"{a:04X}={v}" for a, v in zip(addresses, combo)) + f" @{at}"
            if what in exclude or reach == avoid:
                continue
            key = (-reach, sum(abs(v - c) for v, c in zip(combo, now)))
            if best is None or key < best[0]:
                best = (key, reach, combo, what)
        if best and best[1] >= lost + MARGIN:
            _, reach, combo, what = best
            found = Candidate(what, masks=masks, first=at, prior=0.0, reach=reach)
            found.writes = _with(search.writes, at, zip(addresses, combo))
            return found
    return None


def repair(search: Search, masks: np.ndarray, held_p: np.ndarray, onset_p: np.ndarray,
           lost: int, limit: int, max_edits: int, max_backtracks: int, width: int = 4,
           depth: int = 12, window: int = 300, max_window: int = 2400, cap: int = 400,
           stuck_budget: float | None = None, tiers=(), workers: int = 4):
    """Greedy repair with backtracking: (masks, video frame sync is lost at, edits).

    Each step takes the best set of edits at the current loss of sync
    (`beam`), widening the window up to `max_window` when none helps. When
    still none helps, the most recent sets are undone one at a time and the
    next-best set at that point is tried instead -- a weak early fix can put
    the emulator on a wrong track that only shows later. The
    furthest-reaching labels found are returned.
    A stuck point that takes more than `stuck_budget` CPU seconds, searching
    and backtracking, ends the run: a wall no edit passes (the tile
    animation at 13,048) is a stop, not hours of search.
    """
    def size(tier):
        return int(np.prod([len(values) for _, values in tier]))

    cheap = tuple(t for t in tiers if size(t) <= IN_PROCESS)
    costly = tuple(t for t in tiers if size(t) > IN_PROCESS)

    def find(m, at, exclude, avoid=None):
        # Cheapest first: the narrow beam, then a small resync (the tile tier
        # is 168 replays; widening to 2,400 frames can take half an hour),
        # then the wider beams, then a large resync.
        w = window
        while True:
            found = beam(search, m, held_p, onset_p, at, limit, width, depth, w, cap,
                         exclude, avoid)
            if found is not None or search.out_of_time():
                return found
            if w == window and cheap:
                found = resync(search, m, at, limit, cheap, exclude, avoid, workers)
                if found is not None or search.out_of_time():
                    return found
            if w >= max_window:
                return resync(search, m, at, limit, costly, exclude, avoid, workers) if costly else None
            w = min(2 * w, max_window)
            print(f"  nothing within {w // 2} frames: widening to {w}", flush=True)

    # stack entries: (masks before the edit, sync lost before it, edit, tried
    # edits, state writes before it)
    stack: list[tuple[np.ndarray, int, Candidate, set, dict]] = []
    best = (masks, lost, [])
    search.best_writes = dict(search.writes)
    backtracks = 0
    while lost < limit and len(stack) < max_edits:
        t, cpu = time.time(), time.process_time()
        search.deadline = None if stuck_budget is None else search.clock() + stuck_budget
        winner = find(masks, lost, frozenset())
        tried: set = set()
        if winner is None:
            # Undo edits until one has an untried alternative that helps.
            undone_first = lost
            while (stack and winner is None and backtracks < max_backtracks
                   and not search.out_of_time()):
                masks, lost, undone, tried, search.writes = stack.pop()
                undone_first = min(undone_first, undone.first)
                tried.add(undone.what)
                backtracks += 1
                search.base(masks, undone_first)
                winner = find(masks, lost, frozenset(tried), undone.reach)
                print(f"  backtrack: undo {undone.what} -> "
                      f"{winner.what if winner else 'no alternative'}", flush=True)
            if winner is None:
                why = (f"over its {stuck_budget:.0f} CPU-s budget" if search.out_of_time()
                       else f"no set of edits plays {MARGIN} frames longer")
                print(f"stuck at video frame {best[1]}: {why}", flush=True)
                break
            first = min(winner.first, undone_first, lost)
        else:
            first = min(winner.first, lost)
        stack.append((masks, lost, winner, tried, dict(search.writes)))
        masks = winner.masks
        if winner.writes is not None:
            search.writes = winner.writes
        new_lost = search.base(masks, first)
        # CPU seconds as well: at nice 19 on battery, wall-clock time is
        # mostly time the job was not run.
        print(f"{winner.what:32s} sync {lost} -> {new_lost}  ({time.time() - t:.0f}s, "
              f"{time.process_time() - cpu:.0f} cpu-s, {search.played} frames played)", flush=True)
        lost = new_lost
        if lost > best[1]:
            best = (masks, lost, [{"edit": e.what, "at": e.first, "lost_before": lb,
                                   "prior": round(e.prior, 3)} for _, lb, e, _, _ in stack])
            search.best_writes = dict(search.writes)
    return best


def fresh_reach(make_replayer, masks: np.ndarray, first: int, limit: int, judge_from: int,
                run: int = RUN, writes: dict | None = None) -> int:
    """Where a new emulator (`make_replayer()`), replaying `masks` from the
    anchor under the search's own rule, first loses sync. The search's
    chain of save states is bookkeeping, and PyBoy's save states have lost
    state before (the input latch): what is written as verified is this."""
    replayer = make_replayer()
    search = Search(replayer, replayer.save(), limit, run, judge_from)
    search.writes = dict(writes or {})
    return search.play(replayer.save(), masks, limit)


def should_write(labels, first: int, lost: int) -> bool:
    """False when the labels file already holds a verified reach from
    `first` -- buttons alone or given state writes -- at least as long as
    `lost`: a weaker run must not overwrite it."""
    if "verified" not in labels:
        return True
    held = labels["verified"][first:]
    if "verified_resync" in labels:
        held = held | labels["verified_resync"][first:]
    existing = first + (int(np.argmin(held)) if not held.all() else len(held))
    return lost > existing


def write_repair(labels, key: str, masks: np.ndarray, first: int, lost: int, limit: int,
                 name: str, edits: int, writes: dict | None = None) -> None:
    """Store a run's buttons and state writes over everything it replayed,
    [first, limit), and mark what it verified -- and only that: from `first`
    on, older claims and writes are cleared, so a rerun under a stricter
    check cannot inherit an earlier run's further reach.
    Frames before the run's first write are `verified` (buttons alone);
    from it to `lost`, `verified_resync` (buttons given the writes)."""
    n = len(labels["buttons"])
    if "buttons_repaired" not in labels:
        labels["buttons_repaired"] = labels[key][:]
    for flag in ("verified", "verified_resync"):
        if flag not in labels:
            labels[flag] = np.zeros(n, bool)
    labels["buttons_repaired"][first:limit] = masks[first:limit].astype(np.uint8)
    mine = {f: p for f, p in (writes or {}).items() if first <= f < limit}
    split = min(min(mine, default=lost), lost)
    verified, resync = labels["verified"][:], labels["verified_resync"][:]
    # The replay is chained: once this run's buttons differ from `first` on,
    # nothing after it stays verified, whatever an older run reached there.
    verified[first:] = False
    resync[first:] = False
    verified[first:split] = True
    resync[split:lost] = True
    labels["verified"][:], labels["verified_resync"][:] = verified, resync
    kept = labels["state_writes"][:] if "state_writes" in labels else np.zeros((0, 3), np.int32)
    kept = kept[kept[:, 0] < first]
    rows = np.concatenate([kept, rows_from_writes(mine)]) if len(mine) else kept
    rows = rows[np.argsort(rows[:, 0], kind="stable")]
    if "state_writes" in labels:
        del labels["state_writes"]
    labels["state_writes"] = rows.astype(np.int32)
    labels["verified"].attrs[f"repair_{name}"] = json.dumps(
        {"start": first, "until": lost, "edits": edits, "first_write": split if mine else None})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", default="data/longplay/video.h5")
    parser.add_argument("--labels", default="data/longplay/labels.h5")
    parser.add_argument("--frames", type=int, default=20_000, help="video frames to repair")
    parser.add_argument("--anchor-state", default=None)
    parser.add_argument("--anchor-search", type=int, nargs=2, default=(6000, 7600))
    parser.add_argument("--buttons", default="auto",
                        help="labels to start from: auto = buttons_ruled when present, else buttons")
    parser.add_argument("--max-edits", type=int, default=200)
    parser.add_argument("--max-backtracks", type=int, default=12,
                        help="times to undo recent edits and try the next-best one when stuck")
    parser.add_argument("--settle", type=int, default=150,
                        help="video frames after the start that are not judged (boot screens)")
    parser.add_argument("--width", type=int, default=4, help="beam width")
    parser.add_argument("--depth", type=int, default=12, help="beam rounds")
    parser.add_argument("--window", type=int, default=300,
                        help="frames before the loss of sync that edits may touch")
    parser.add_argument("--max-window", type=int, default=2400)
    parser.add_argument("--cap", type=int, default=400, help="edits tried per member per round")
    parser.add_argument("--resync", default="tile,random",
                        help="hidden-state tiers to try where no edit helps: tile, random, or none")
    parser.add_argument("--stuck-budget", type=float, default=None,
                        help="CPU seconds a stuck point may take before the run stops there")
    parser.add_argument("--only-if-better", action="store_true",
                        help="write only if this run verifies further than the file already does")
    parser.add_argument("--name", default="poweron")
    parser.add_argument("--dry-run", action="store_true", help="do not write the labels file")
    args = parser.parse_args()

    started = time.time()
    with h5py.File(args.video) as h5, h5py.File(args.labels) as labels:
        snapper = Snapper(h5["colours"][:])
        key = args.buttons if args.buttons != "auto" else (
            "buttons_ruled" if "buttons_ruled" in labels else "buttons")
        masks = labels[key][:].astype(np.int64)
        # Labels this repair wrote before carry the state writes they need.
        start_writes = (writes_from_rows(labels["state_writes"][:])
                        if key == "buttons_repaired" and "state_writes" in labels else {})
        print(f"starting from {key}" + (f" with {len(start_writes)} state writes"
                                         if start_writes else ""), flush=True)
        onset_p = labels["onset_p"][:].astype(np.float32)
        held_p = labels["held_p"][:].astype(np.float32)
        env = None
        if args.anchor_state:
            env = anchored(args.anchor_state)
            pyboy = env.pyboy
            anchor, share = find_anchor(env, h5, snapper, *args.anchor_search)
            offset = anchor + 1
            masks = release_carried(masks, offset)
            first_frame = offset
            print(f"anchor {args.anchor_state} = video frame {anchor} ({share:.4f})", flush=True)
        else:
            offset, _ = align(h5, snapper)
            pyboy = boot()
            first_frame = 0
        limit = first_frame + args.frames
        # Only the frames this run judges (with room for the offset tracker
        # and the text check's slack before the start).
        start = max(first_frame - 64, 0)
        video = VideoWindow(h5, start, limit + 8 - start)
        # Only what the run replays: every candidate's masks are this long.
        masks, onset_p, held_p = masks[:limit + 8], onset_p[:limit + 8], held_p[:limit + 8]
    original = masks.copy()
    replayer = Replayer(pyboy, snapper, video, masks, offset)
    search = Search(replayer, replayer.save(), limit,
                    judge_from=first_frame + (0 if args.anchor_state else args.settle))
    search.writes = dict(start_writes)
    lost = search.base(masks, first_frame)
    print(f"labels as decoded: in sync until video frame {lost}", flush=True)
    tiers = tuple({"tile": TILE_ANIMATION, "random": RANDOM}[t]
                  for t in args.resync.split(",") if t in ("tile", "random"))
    masks, lost, edits = repair(search, masks, held_p, onset_p, lost, limit, args.max_edits,
                                args.max_backtracks, args.width, args.depth, args.window,
                                args.max_window, args.cap, args.stuck_budget, tiers)
    writes = search.best_writes
    if env:
        env.close()
    else:
        pyboy.stop(save=False)

    judge_from = first_frame + (0 if args.anchor_state else args.settle)
    fresh_envs = []

    def new_replayer():
        if args.anchor_state:
            fresh_envs.append(anchored(args.anchor_state))
            return Replayer(fresh_envs[-1].pyboy, snapper, video, masks, offset)
        fresh_envs.append(boot())
        return Replayer(fresh_envs[-1], snapper, video, masks, offset)

    searched = int(lost)
    lost = min(searched, fresh_reach(new_replayer, masks, first_frame, limit, judge_from,
                                     writes=writes))
    for made in fresh_envs:
        made.close() if args.anchor_state else made.stop(save=False)
    print(f"fresh replay of the result: in sync until {lost} (search: {searched})", flush=True)

    changed = int((masks != original).sum())
    report = {"name": args.name, "start": first_frame, "verified_until": int(lost),
              "search_reach": searched, "state_writes": rows_from_writes(writes).tolist(),
              "limit": limit, "edits": edits, "frames_changed": changed,
              "anchor_state": args.anchor_state, "seconds": round(time.time() - started)}
    print(f"verified video frames {first_frame}-{lost} with {len(edits)} edits "
          f"({changed} frames changed) in {report['seconds']}s", flush=True)
    Path(f"results/repair_{args.name}.json").write_text(json.dumps(report, indent=1))
    if not args.dry_run:
        with h5py.File(args.labels, "a") as labels:
            if args.only_if_better and not should_write(labels, first_frame, int(lost)):
                print("not written: the labels file already verifies further", flush=True)
            else:
                write_repair(labels, key, masks, first_frame, int(lost), limit, args.name,
                             len(edits), writes)


if __name__ == "__main__":
    main()
