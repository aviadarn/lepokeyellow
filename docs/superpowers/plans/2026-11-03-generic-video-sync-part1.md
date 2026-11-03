# Generic video sync, Part 1 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the special-case label repair with a generic, emulator-judged
search that, from power-on and v3's labels alone, re-finds today's fixes and
plays through the 2:32 conversation with the girl. Also measure what
randomness matching needs (the Part 2 spike).

**Architecture:**
- **Edit grammar.** A generic grammar (`scripts/edits.py`) proposes edits to
  press intervals of any button, ranked by the IDM's likelihood.
- **Beam search.** A beam over edit sets (`scripts/repair_labels.py`)
  replays candidates from snapshots under the unchanged strict judge.
- **Faster judge.** The judge gets faster with identical verdicts
  (`scripts/sync_opening.py`).
- **Randomness spike.** A small random-number tool (`scripts/rng_sync.py`)
  answers whether one resync of `hRandomAdd/hRandomSub` keeps the emulator in
  step.

**Tech Stack:** Python 3.13, PyBoy 2.8.1 (CGB), numpy, h5py, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-11-03-generic-video-sync-design.md`

## Global Constraints

- **The judge is fixed:**
  - `in_sync` is share ≥ `SYNC_SHARE` = 0.98 and text < `TEXT_LIMIT` = 6.
  - `TEXT_PATCH` = 8, `TEXT_SLACK` = 12, `RUN` = 20.
  - Any change keeps 0 false alarms on the synthetic set
    (`results/text_check_synth.json`, 1,260,000 frames).
- **Verified means a fresh replay agrees.** Frames are marked verified only
  when a fresh replay from the anchor agrees. Search bookkeeping never counts.
- **Mac safety:**
  - nice 19, at most 4 worker processes, one heavy job at a time;
  - the memory watchdog (`scratchpad/watchdog_strict.sh` pattern), with long
    runs in `screen -dmS`.
- **Nothing derived from the ROM, a state or the video is committed.**
- **Interfaces stay:**
  - `labels.h5` `buttons_repaired`/`verified`, written only through
    `write_repair`;
  - `scripts/export_video.py` reads them unchanged.
- **Starting labels:** v3 `buttons_ruled`, with v3 `held_p` and `onset_p`
  from `data/longplay/labels.h5`.
- **Window:** W = 300, doubling up to 2,400 when a beam finds nothing.
- **Beam:** B = 4, D = 12 to start, about 400 candidates per member per round.
- **Speed targets:** ≤ 0.5 ms per judged frame; a stuck point takes ≤ 5 min
  (median) and ≤ 30 min (90th percentile); the Part 1 run finishes in one
  night (≤ 8 h).
- **Commits:** go on branch `tight-verify`, linear. End messages with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **A press that starts before the window and ends inside it.** Edits to it
   must stay valid intervals and set `first` to the earliest changed frame.
   Owned by Task 4.
2. **A beam node whose earlier edits lie before the original loss.** Its
   children replay from a snapshot at or before
   `min(node.first, child.first, lost)`. Owned by Task 5.
3. **Edits at the ends of the labels.** Inserts or moves that would run past
   frame N or before 0 are clipped, never wrapped. Owned by Task 4.
4. **The fast judge on a frame that fails only on text for exactly RUN
   frames.** It gives the same verdicts as the reference. Owned by Task 3.
5. **A run that verifies less than the labels file already holds** (9,116).
   It must not overwrite it. Owned by Task 5.

---

### Task 1: Base commit of the strict check

The strict check, direction edits, chains, the snapshot fix and their tests
are uncommitted on `tight-verify`, and 554 tests pass. This plan builds on
them.

- [ ] **Step 1:** Run `uv run pytest -q 2>&1 | tail -1`. Expected:
  `554 passed`.
- [ ] **Step 2:** Commit:
  - the files: `scripts/sync_opening.py`, `scripts/repair_labels.py`,
    `tests/test_text_check.py`, `tests/test_repair.py`, `README.md`,
    `docs/longplay_labels.md`, `docs/superpowers/specs/2026-11-03-generic-video-sync-design.md`,
    this plan, `results/repair_poweron_strict.json`,
    `results/sync_opening_strict.json`, `results/text_check_synth.json`,
    `results/repair_strict.log`;
  - the message: "Strict text check: wrong letters and cursors break sync;
    repair judged from before the loss".

### Task 2: Why the girl's text prints behind the video (diagnosis, no production code)

**Files:**
- Create: `results/girl_text_speed.json`.
- Scratch scripts go in the scratchpad, not the repo.

**Interfaces:**
- Consumes:
  - `scripts/sync_opening.Replayer`, `in_sync`;
  - the hand fix from the 2:32 investigation: `buttons_repaired` with DOWN
    released at 9,110, RIGHT cut to 9,120–9,126, A tap at 9,144.
- Produces:
  - `results/girl_text_speed.json` with
    `{"cause": str, "is_button": bool, "evidence": [...], "conversation_end": int}`.
    `conversation_end` is the video frame at which the girl's last text box
    closes. Task 6 uses it.

- [ ] **Step 1: Measure the print rate.** In video and emulator, measure
  letters printed per frame:
  - for the girl's text (video 9,143 to 9,400);
  - for one verified dialogue (mom, about 8,500–8,800).

  Expected: the numbers show whether the video prints faster everywhere or
  only at the girl.
- [ ] **Step 2: Read the text speed.** Read `wOptions` (0xD354, low nibble =
  text delay) in the emulator at 9,140. Compare it with the delay the
  video's print rate implies.
- [ ] **Step 3: Test the button explanations.** Try B held (5, 20 and 60
  frames, from 9,140 to 9,150), and A+B together, on the hand fix. Record
  the first loss for each.
- [ ] **Step 4: Rule and record.**
  - Write `results/girl_text_speed.json`.
  - If `is_button`, Task 5's grammar must be able to express it: B and A
    holds of 20/60 are in the grammar.
  - If not, write a ledger ruling. The Part 1 test becomes "an explained stop
    at the girl", and Task 6 Step 4's threshold changes accordingly.

### Task 3: Fast judge with identical verdicts

**Files:**
- Modify: `scripts/sync_opening.py`:
  - `text_mismatch` (around lines 89–97);
  - `text_mismatch_near` (99–108);
  - `in_sync` (111–112);
  - `Replayer.step` (215–236);
  - `replay` (252–289).
- Test: `tests/test_text_check.py`.

**Interfaces:**
- Produces:
  - `Replayer.step()` → `(v, share, text)`, where `text` is `None` when the
    share already failed (not judged), else an `int`;
  - `in_sync(share, text)` → `share >= SYNC_SHARE and text is not None and text < TEXT_LIMIT`;
  - `text_mismatch_near` may return any value `< TEXT_LIMIT` as soon as one
    neighbour is under the limit. Values ≥ `TEXT_LIMIT` stay the minimum
    over the neighbours;
  - `replay()` → `text_mismatch_frames` counts only frames that pass the
    share and fail on text.

- [ ] **Step 1: Write failing tests.**
  - `test_fast_text_check_agrees_with_the_full_one`: on 200 random frame
    pairs built from `screen()` glyphs (random letters, positions, ink
    shades, art colours, unsnapped 255 pixels; seed 0), assert
    `(text_mismatch(a, b) >= TEXT_LIMIT) == (reference(a, b) >= TEXT_LIMIT)`.
    `reference` is the current implementation, copied into the test file as
    `_reference_text_mismatch`.
  - `test_near_stops_at_the_first_neighbour_under_the_limit`: a monkeypatched
    `text_mismatch` that counts calls. It is called at most 2 times when
    `video[v+1]` matches.
  - `test_text_is_not_judged_when_the_share_failed`: on a fake `pyboy`/video
    pair with share 0.5, `Replayer.step()[2] is None` and
    `in_sync(0.5, None) is False`.
- [ ] **Step 2:** Run
  `uv run pytest tests/test_text_check.py -q`. Expected: the 3 new tests
  FAIL.
- [ ] **Step 3: Implement.**
  - In `text_mismatch`, when the paper/ink pixels that differ in both frames
    number fewer than `TEXT_LIMIT` in total, return that count without
    computing the patch sums.
  - In `text_mismatch_near`, scan neighbours in order of `|j|` and return
    the first value under the limit.
  - In `Replayer`, keep a per-video-frame cache of `classes[video[u]]`,
    bounded to 64 entries.
  - `Replayer.step` skips the text check when `scores[0] < SYNC_SHARE`.
- [ ] **Step 4:** Run `uv run pytest -q`. Expected: all pass. Then confirm
  0 false alarms with the synthetic script
  (`scratchpad/falsealarm/synth_false_alarms.py`). Expected output:
  `'text_losses': 0`.
- [ ] **Step 5: The real-data check.** Run
  `uv run python -m scripts.sync_opening --buttons buttons_repaired --frames 12000 --out /tmp/x.json`
  (the scratchpad is fine). Expected:
  - `lost_at_video_frame` 9116, `offset_moves` 15 and `in_sync_share`
    0.6974…, the same as `results/sync_opening_strict.json`;
  - a run time ≤ 0.5 ms × 11,931 frames ≈ 6 s for the replay part, plus the
    intro alignment.
- [ ] **Step 6:** Commit "Faster strict judge, same verdicts".

### Task 4: Generic edit grammar and IDM prior

**Files:**
- Create: `scripts/edits.py`.
- Test: `tests/test_edits.py`.

**Interfaces:**
- Consumes: `runs`, `_set`, `_peaks` and `Candidate` from
  `scripts/repair_labels.py` (moved into `scripts/edits.py` and re-exported
  from `repair_labels`, so existing imports keep working).
- Produces:
  - `OFFSETS = (1, 2, 4, 8, 16, 32, 64, 128, 256)`, applied with both signs;
  - `LENGTHS = (TAP_FRAMES, 20, 60)`;
  - `GRID = 6`;
  - `prior(masks: np.ndarray, edited: np.ndarray, held_p: np.ndarray, lo: int, hi: int) -> float`.
    It returns the sum over changed (frame, button) of `log q_new − log q_old`,
    where `q = clip(held_p, 0.01, 0.99)` if the bit is set, else
    `1 − clip(...)`;
  - `candidates(masks, held_p, onset_p, lost, window) -> list[Candidate]`.
    Each candidate's `.first` is the earliest changed frame, and `.prior`
    comes from `prior`;
  - `screen(cands: list[Candidate], cap: int) -> list[Candidate]`. It keeps
    the top `max(2, cap // 48)` of each (kind, button) group by prior, then
    fills up to `cap` by global prior. A kind is the first word of `.what`.

  Edit names, one per kind:

  | kind | name |
  |---|---|
  | insert | `add {button} {t}` or `add {button} {t}x{len}` |
  | delete | `drop {button} {s}-{e}` |
  | move start | `start {button} {s}->{s2}` |
  | move end | `end {button} {e}->{e2}` |
  | set length | `length {button} {s}-{e}->{len}` |
  | split | `split {button} {s}-{e}@{t}` |
  | change button | `swap {button}->{other} {s}-{e}` |

- [ ] **Step 1: Write failing tests**, in `tests/test_edits.py`.
  - `test_every_kind_is_proposed_for_every_button`: masks with one 10-frame
    press of each button inside the window. Every kind appears for every
    button.
  - `test_a_long_hold_can_become_a_tap`: masks with RIGHT held 9,120–9,277,
    `lost` = 9,143. The candidates include
    `length right 9120-9277->{TAP_FRAMES}`, whose masks equal RIGHT on
    9,120–9,125 only.
  - `test_presses_starting_before_the_window_are_editable`: a DOWN press from
    `lost − 400` to `lost − 100`, with `window` = 300. `end down` edits
    exist, and `.first` is the edited end.
  - `test_edits_are_clipped_at_both_ends_of_the_labels`: presses at frame 0
    and at N − 2, with `window` ≥ N. No candidate has masks of a different
    length, and no index is negative.
  - `test_prior_prefers_what_the_model_believes`: `held_p` of 0.9 on frames
    100–105 for A. `add a 100` has a higher prior than `add b 100`.
  - `test_screen_keeps_every_kind`: with `cap` = 48 from 1,000 candidates,
    every (kind, button) group present before is present after.
- [ ] **Step 2:** Run `uv run pytest tests/test_edits.py -q`. Expected:
  FAIL (no module `scripts.edits`).
- [ ] **Step 3: Implement** `scripts/edits.py` with the interfaces above.
  - Window: `[lost − window, lost + 5)`.
  - Inserts: every button, every `t` on the `GRID` in the window plus the
    top 6 `onset_p` peaks, every length in `LENGTHS`. Skip a `t` where the
    button is already held.
  - Presses: those overlapping the window get move start/end (both signs,
    every offset), set length (each of `LENGTHS`), split (at the top 3
    `onset_p` peaks inside the press, plus its midpoint, as a 2-frame gap
    `[t − 2, t)`) and change button (to each other button).
- [ ] **Step 4:** Run `uv run pytest tests/test_edits.py tests/test_repair.py -q`.
  Expected: PASS. Existing repair tests still import `runs`, `_set` and
  `Candidate`.
- [ ] **Step 5:** Commit "Generic edit grammar over press intervals, ranked
  by the IDM".

### Task 5: Beam search over edit sets replaces near/far/chain

**Files:**
- Modify: `scripts/repair_labels.py`:
  - `Search` gains `reach`;
  - add `beam`;
  - `repair()` uses `beam`;
  - remove `near_candidates`, `far_candidates` and `chain`;
  - `main` gets new flags.
- Test: `tests/test_repair.py`, rewriting the tests that name removed
  functions.

**Interfaces:**
- Consumes: `candidates` and `screen` from Task 4; `Replayer.step` from
  Task 3.
- Produces:
  - `Search.reach(masks: np.ndarray, first: int, lost: int, cap: int) -> int`.
    It plays from `latest(self.snapshots, min(first, lost))`;
  - `beam(search, masks, held_p, onset_p, lost, limit, width=4, depth=12, window=300, cap=400, exclude=frozenset(), avoid=None) -> Candidate | None`.
    `.what` is the edits joined by `" + "`, and `.first` is the minimum over
    the set.
  - `repair(search, masks, held_p, onset_p, lost, limit, max_edits, max_backtracks)`
    keeps its return value `(masks, lost, edits)`;
  - CLI flags: `--width 4 --depth 12 --window 300 --max-window 2400 --cap 400 --only-if-better`.
  - `should_write(labels, first: int, lost: int) -> bool`. It is False when
    `labels["verified"]` already holds a longer reach from `first`. `main`
    calls it under `--only-if-better`.

  The beam's behaviour:
  - Each member expands at its own loss: `candidates(member.masks, …, member.reach, window)`,
    then `screen(…, cap)`.
  - Children replay with `first = min(member.first, child.first)`.
  - It keeps the `width` best, by reach descending, then prior descending,
    then fewer edits. Identical masks are dropped.
  - It returns at once on a child with reach ≥ `min(limit, lost + HORIZON)`.
  - After `depth` rounds it returns the best if its reach ≥ `lost + MARGIN`,
    else `None`.
  - In `repair`, a `None` doubles `window` up to `--max-window` before
    backtracking.

- [ ] **Step 1: Write failing tests**, in `tests/test_repair.py`. Use the
  existing `_fake_search`, where reach is the first differing frame + RUN.
  - `test_beam_turns_a_long_hold_into_a_tap_and_adds_the_missed_a`:
    - truth: DOWN 9,080–9,110, RIGHT 9,120–9,125, A 9,140–9,145;
    - decoded: DOWN to 9,115, RIGHT 9,120–9,277, no A;
    - `onset_p[9140, a] = 0.3`.

    `repair` reaches n and `masks == truth`.
  - `test_beam_finds_three_taps_closer_than_the_margin`: the existing
    three-taps test, asserting only `lost == n` and masks.
  - `test_a_false_press_far_back_is_found_by_widening_the_window`: a false A
    at 200 whose fake effect starts at 1,700, i.e. reach = 1,700 + RUN when
    that A is present. `repair` reaches n, with the A dropped.
  - `test_children_replay_from_before_the_original_loss`: a `_FixedLoss`-style
    replayer, where frames 105–134 are bad whatever is pressed. `beam` on it
    returns `None`. Then Review Focus 2: a member with an edit at 90 makes
    the child's replay start at a snapshot ≤ 90, asserted by recording
    `load` arguments.
  - `test_only_if_better_keeps_a_further_verified_reach` (Review Focus 5):
    `write_repair(..., lost=5_000)` on a file whose `verified` already
    reaches 9,116, behind the `--only-if-better` guard function
    `should_write(labels, first, lost) -> bool`. It returns False and the
    file is unchanged.
  - Delete `test_near_edits_drop_move_and_add`,
    `test_far_edits_only_drop_taps_further_back` and
    `test_near_edits_add_and_swap_directions`. Task 4's tests replace them.
  - Change `test_repair_adds_missed_taps_and_drops_false_ones` and
    `test_repair_chains_*` to assert `lost` and masks, not edit names. They
    pass `held_p = np.full((n, 8), 0.5)` to the new `repair` signature.
- [ ] **Step 2:** Run `uv run pytest tests/test_repair.py -q`. Expected: the
  new tests FAIL (`beam`, `Search.reach` and `should_write` not defined).
- [ ] **Step 3: Implement** the interfaces above. Remove the old candidate
  functions and the `NEAR`, `FAR`, `DIR_NEAR`, `CHAIN_*`, `TAPS` and
  `DIRECTIONS` constants if nothing else uses them. Update the module
  docstring's "Search" paragraph to describe the grammar and the beam.
- [ ] **Step 4:** Run `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5: Smoke run.**
  - Command:
    `nice -n 19 uv run python -m scripts.repair_labels --frames 4000 --dry-run --name smoke`.
  - Expected:
    - it reaches the options menu (≥ 3,270) and logs beam edits;
    - the seconds per stuck point print in the log;
    - the median is ≤ 300 s.
- [ ] **Step 6:** Commit "Beam search over generic edits replaces the
  special-case repair".

### Task 6: The Part 1 run from power-on

**Files:**
- Create: `results/repair_poweron_generic.json`, `results/sync_opening_generic.json`.
- Modify: `docs/longplay_labels.md` (the repair paragraph and Reproduce),
  `README.md` (row 3b, if the verified reach changes).

**Interfaces:**
- Consumes: Task 5's CLI; `conversation_end` from Task 2.
- Produces: `labels.h5` `buttons_repaired`/`verified`, written only if
  better.

- [ ] **Step 1: Back up** `buttons_repaired` and `verified` to the
  scratchpad (an npz, as before).
- [ ] **Step 2: Start the run** in `screen` with the watchdog:
  `repair_labels --name poweron_generic --frames 18000 --only-if-better`.
  The default starting labels are `buttons_ruled`. The run is 5:00 of video,
  long enough to reach the rival battle for Task 7. Expected: the log shows
  the beam's edits. The memory watchdog stays quiet.
- [ ] **Step 3: Check it with a fresh replay.**
  `scripts.sync_opening --buttons buttons_repaired --frames 18000 --out results/sync_opening_generic.json`.
  Expected: `lost_at_video_frame` matches the run's `verified_until`, within
  the 60-frame vs 20-frame loss rules (≤ 40 frames).
- [ ] **Step 4: Judge it against the spec.**
  - Success: `verified_until` ≥ `conversation_end` + 600, and ≥ 9,116, with
    no hand edits.
  - If it falls short, record where and why (stop frame, the frames
    rendered), and ledger the result. Don't tune for this case.
- [ ] **Step 5: Re-export and update the docs.**
  - Run `scripts.export_video`, then update `docs/longplay_labels.md` (the
    verified reach, the edit count, and the Reproduce line with the new
    flags) and `README.md`.
  - Commit "Generic repair from power-on: …" with the reach in the message.

### Task 7: Randomness spike

**Files:**
- Create: `scripts/rng_sync.py`, `tests/test_rng_sync.py`,
  `results/rng_spike.json`.
- Modify: the spec's Part 2 section (the decision).

**Interfaces:**
- Produces:
  - `RANDOM_ADD = 0xFFD3`, `RANDOM_SUB = 0xFFD4`;
  - `set_rng(pyboy, add: int, sub: int) -> None`;
  - `find_rng(replayer, snapshot, horizon: int, values=None, workers: int = 4) -> list[tuple[int, int]]`.
    It returns every `(add, sub)` that, written at the snapshot, keeps all
    of the next `horizon` frames `in_sync`.

- [ ] **Step 1: Write the failing test**
  `test_find_rng_recovers_the_random_state_that_made_a_recording`. It skips
  without the ROM or `states/cgb/`.
  - From a CGB state with a wandering NPC, run 300 frames with the RNG set
    to (77, 150), recording the frames as the "video".
  - From the same state, `find_rng` over `values` = 64 pairs including
    (77, 150) must include (77, 150).
- [ ] **Step 2:** Run `uv run pytest tests/test_rng_sync.py -q`. Expected:
  FAIL (no module).
- [ ] **Step 3: Implement.** `find_rng` splits `values` (default: all
  65,536) across `workers` processes at nice 19. Each worker has its own
  PyBoy loaded from the snapshot blob.
- [ ] **Step 4:** Run `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5: Measure on the real video.**
  - At Task 6's stop: if its frames show only sprites or battle numbers
    differing, run `find_rng` there (horizon 30). Record the matching values,
    the sweep time, and how many frames stay in sync after the resync with
    the button search continuing.
  - If Task 6 passed every random event up to 5:00 without a resync, record
    that instead: it means the two emulators' random numbers agree.
- [ ] **Step 6: Decide and commit.**
  - Write `results/rng_spike.json` and the decision into the spec's Part 2:
    per-event resync, timer-phase search, or none needed.
  - Commit "Randomness spike: …".
