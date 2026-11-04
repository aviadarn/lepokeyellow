# Generic video sync, Part 2 — Hidden-state resync

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the repair pass walls no button explains: where the two
emulators' hidden state parts (tile animation counters, random bytes), the
search writes the bytes that make the replay match the video, records each
write, and every replay — search, fresh check, `sync_opening` — applies it.
Then the repair continues through Oak's lab toward the rival battle.

**Architecture:**
- **State writes.** Each write is a (video frame, address, value) triple.
  The Replayer applies it just before the tick that draws that frame.
- **Storage.** Writes are stored in `labels.h5`. Frames after the first
  write are `verified_resync`, never `verified`.
- **Search.** When the beam finds nothing even at the widest window, the
  repair tries resync tiers before backtracking:
  - tile animation (21 × 8 values, in-process);
  - random bytes (65,536 values, on 4 workers).

**Tech Stack:** Python 3.13, PyBoy 2.8.1, numpy, h5py, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-11-03-generic-video-sync-design.md`
(Part 2, with its spike result).

## Global Constraints

- **The judge is unchanged.** `in_sync`: share ≥ 0.98, text < 6,
  `TEXT_SLACK` 12, `RUN` 20, tile slack only when SCX/SCY are on the grid.
- **`verified` stays buttons-only:** frames from the anchor up to the first
  write.
  - Frames after the first write, up to the loss, are `verified_resync`.
  - Both come from a fresh replay (`fresh_reach`) that applies the same
    writes.
- **Every write is recorded:** in the report, and in `labels.h5`
  `state_writes` ((M, 3) int32: frame, address, value).
- **Tile-animation tier:** `0xFFD8` in 0..20 and `0xD084` in 0..7.
- **Random tier:** `0xFFD3` and `0xFFD4` in 0..255 each.
- **Writes land 40 frames before the loss** (`RESYNC_LEAD`).
- **A write is accepted** when it plays MARGIN frames past the loss. Ties
  go to the smallest change from the bytes' current values.
- **Mac safety:**
  - nice 19, at most 4 workers, the watchdog with its battery guard;
  - `--stuck-budget` covers the resync too.
- **Commits** go on `tight-verify`, ending with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **A write at frame w must not apply when replaying from a snapshot after
   w.** It also must not be applied twice when a snapshot sits exactly at
   w. Owned by Task 1.
2. **Backtracking past a committed write** removes the write as well as the
   masks. Owned by Task 2.
3. **The fresh replay and `sync_opening` apply the same writes** as the
   search; without them they must lose sync at the wall. Owned by Tasks 1
   and 3.
4. **A later run without writes over a stretch** clears that stretch's old
   writes and `verified_resync`. Owned by Task 3.
5. **The export marks frames after a write as resync-verified, not plain
   verified** (`verified_mode`). Owned by Task 3.

---

### Task 1: Replayer applies recorded state writes

**Files:**
- Modify: `scripts/sync_opening.py`:
  - `Replayer` gains `writes`, applied in `step`;
  - `replay()` and `main` read `state_writes` from the labels file.
- Test: `tests/test_state_writes.py`.

**Interfaces:**
- Produces:
  - `Replayer.writes: dict[int, tuple[tuple[int, int], ...]]`, mapping a
    video frame to its (address, value) pairs, applied in `step` after
    pressing the buttons and before `tick`;
  - `writes_from_rows(rows: np.ndarray) -> dict`;
  - `rows_from_writes(writes: dict) -> np.ndarray`, int32 (M, 3), sorted.
  - `replay(...)` takes `writes=None`. `main` reads
    `labels["state_writes"]` when present.

- [ ] **Step 1: Failing tests.**
  - `test_writes_round_trip_through_rows`.
  - `test_a_write_applies_once_on_its_frame`: a fake pyboy with a dict
    memory and a tick counter. The write lands exactly when step() reaches
    that video frame, and not again on the next step.
  - `test_a_write_before_the_snapshot_is_not_replayed`: `load` a snapshot
    taken after the write's frame. Stepping on doesn't re-apply it.
- [ ] **Step 2:** Run `uv run pytest tests/test_state_writes.py -q`.
  Expected: FAIL (no functions).
- [ ] **Step 3: Implement.** `step` looks up `self.writes.get(v)` for the
  v it is about to draw. Replaying from a later snapshot never reaches the
  earlier v, which settles Review Focus 1.
- [ ] **Step 4:** Run `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5:** Commit "Replayer applies recorded state writes".

### Task 2: Resync tiers in the repair search

**Files:**
- Modify: `scripts/repair_labels.py`:
  - `Search.play`/`reach` take `writes`;
  - `Search.writes` holds the committed ones;
  - add `resync()`;
  - `repair()` tries it before backtracking and keeps writes on the stack.
- Modify: `scripts/rng_sync.py`. Add
  `find_state(replayer, snapshot, cap, registers, workers, writes)`, which
  returns `[(reach, values)]`: a generic worker version of `find_rng`.
- Test: `tests/test_repair.py`, `tests/test_rng_sync.py`.

**Interfaces:**
- Produces:
  - `TILE_ANIMATION = ((0xFFD8, range(21)), (0xD084, range(8)))`;
  - `RANDOM = ((0xFFD3, range(256)), (0xFFD4, range(256)))`;
  - `RESYNC_LEAD = 40`;
  - `resync(search, masks, lost, limit, tiers) -> Candidate | None`.
    The candidate has `.writes` (the full set, as a dict) and
    `.what = "write FFD8=0 D084=1 @13008"`.
  - Tiers are tried in order:
    - a tier under 1,000 combinations runs in-process through
      `Search.reach`;
    - a larger one runs through `find_state` on 4 workers.

- [ ] **Step 1: Failing tests.**
  - `test_resync_finds_the_write_that_passes_a_wall`: a fake replayer whose
    frames are out of sync from 300 unless memory[0xFFD8] was set to 7 at a
    frame ≤ 300. `repair` with the TILE_ANIMATION tier reaches n, and the
    report names the write.
  - `test_backtracking_removes_a_committed_write`: after an undone write,
    `search.writes` no longer holds it.
  - `test_find_state_ranks_values_by_reach`, in test_rng_sync (skips without
    the ROM): recording with FFD8 = 5, search 0..20; the best reach is
    value 5.
- [ ] **Step 2:** Run `uv run pytest tests/test_repair.py tests/test_rng_sync.py -q`.
  Expected: the new tests FAIL.
- [ ] **Step 3: Implement.**
  - `repair` calls `resync` after the widest window fails and before
    backtracking.
  - A winner is committed like a set of edits: the stack keeps
    (masks, writes).
  - The repair's `--resync tile,random` flag (default `tile,random`) picks
    the tiers.
- [ ] **Step 4:** Run `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5:** Commit "Repair resyncs hidden state at walls no edit
  passes".

### Task 3: Writes in the labels file, `verified_resync`, export `verified_mode`

**Files:**
- Modify: `scripts/repair_labels.py`:
  - `write_repair` stores the writes and `verified_resync`;
  - `fresh_reach` applies the writes;
  - `main` reads them when it starts from `buttons_repaired`.
- Modify: `scripts/export_video.py`. `label_arrays` also returns `resync`.
  PokeDreamer chunks and SCHEMA-3 get a per-step `verified_mode`
  (0 unchecked, 1 verified, 2 verified with state writes).
- Test: `tests/test_repair.py`, `tests/test_export_video.py`.

- [ ] **Step 1: Failing tests.**
  - `test_write_repair_splits_verified_at_the_first_write`: with writes at
    500, `verified` covers [0, 500) and `verified_resync` covers
    [500, lost). The writes' rows are stored, and earlier rows in
    [first, limit) are replaced.
  - `test_a_run_without_writes_clears_old_writes_in_its_stretch` (Review
    Focus 4).
  - `test_fresh_reach_applies_the_writes`: the same fake as Task 2.
  - `test_export_marks_steps_verified_with_writes`: `verified_mode` is 2 on
    those steps.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3: Implement.** For the button priority, the export treats
  `verified | verified_resync` as checked.
- [ ] **Step 4:** Run `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5:** Commit "State writes stored with the labels; resync-verified
  frames marked apart".

### Task 4: The run through Oak's lab

- [ ] **Step 1:** Back up `buttons_repaired`, `verified`,
  `verified_resync` and `state_writes`.
- [ ] **Step 2: Start the run** unattended in `screen` with the watchdog:
  `repair_labels --name poweron_resync --buttons buttons_repaired --frames 24000 --only-if-better --stuck-budget 1800`.
  It starts from the code's own verified labels, which are in sync to
  13,048. Expected: at 13,048 a tile-animation write is found, and the
  repair goes on through the lab.
- [ ] **Step 3: Fresh replay.** Run
  `sync_opening --buttons buttons_repaired --frames 24000 --out results/sync_opening_resync.json`
  (it applies the writes). Expected: in sync past 13,048, consistent with
  the run's reach under its 60-frame rule.
- [ ] **Step 4: Record what stopped it.** Classify the stop: random (battle
  numbers), buttons, or hidden state. If the random tier ran, record its
  sweep time and result.
- [ ] **Step 5: Wrap up.** Re-export, update the docs, the README and the
  spec's Part 2 status, then commit.
