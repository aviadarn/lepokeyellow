# Longplay Labels for the Whole Game — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the longplay's button labels trustworthy across the whole 6 h 53 min video, with every exported step saying how far it is checked.

**Architecture:** Two rules the screen proves — a step's scroll implies its direction was held just before, a `▼` prompt answered implies an A press — are checked and enforced over all 1,485,634 frames (`poke/label_rules.py`, `scripts/check_labels.py`). A v4 model is fine-tuned from v3 on new synthetic play aimed at its weak spots (A in text and battle). Exports prefer emulator-repaired labels, then rule-enforced, then decoded, and carry per-step check counts.

**Tech Stack:** Python ≥ 3.12, uv, numpy, h5py, PyBoy 2.8.1, torch (MPS).

**Spec:** `docs/superpowers/specs/2026-11-02-longplay-full-game-labels-design.md`

## Global Constraints

- The user works on this Mac while jobs run: every long job at `nice -n 19`, ≤ 4 synthetic-play workers (~1 GB each), GPU jobs (~6 GB) one at a time, under a pause-at-25 %-free / resume-at-40 % watchdog in its own `screen` session.
- Button bit order everywhere: `poke.synth.BUTTONS` = up, down, left, right, a, b, start, select.
- Scroll shift `(dy, dx)` means `new[y, x] == old[y+dy, x+dx]`: `(-2,0)` up, `(2,0)` down, `(0,-2)` left, `(0,2)` right.
- Rules never remove a press; they only add.
- No ROM, save state, or video-derived file is committed; tests must pass without the ROM (skip if they need it).
- Results quoted in docs are committed under `results/`.

## Review Focus

- **Scripted walking** (Oak leading the player, being pushed): the screen scrolls with no input. Expected: no press inserted when the model's `held_p` for that direction stays < 0.05 over `t-8 .. t`. → test in Task 2.
- **Edit cuts and fades** between two frames that happen to look shifted. Expected: frames with `cut` or unreadable (`state == 255`) never start a step or a prompt answer. → test in Task 4.
- **Partial-screen motion** (battle slide-ins, menus opening, sprites walking). Expected: not a scroll unless ≥ 97 % of the overlap matches after the shift. → test in Task 1.
- **The `▼` blinking off** while still waiting. Expected: a wait is one span until the text area changes, not one span per blink. → test in Task 3.
- **A held through a prompt from before it appeared** (held-A text speed-up). Expected: a held A carried into the prompt does not count as the answer; a new onset is needed. → test in Task 3.

---

### Task 1: Scroll detection and step starts

**Files:**
- Create: `poke/label_rules.py`
- Test: `tests/test_label_rules.py`

**Interfaces:**
- Produces:
  - `SHIFTS: dict[tuple[int, int], int]` = `{(-2, 0): 0, (2, 0): 1, (0, -2): 2, (0, 2): 3}` (shift → button bit).
  - `scroll(frames: np.ndarray, match: float = 0.97) -> np.ndarray` — `(n,) int8`, the button bit of the direction frame `t` scrolled relative to `t-1`, or `-1`. Frame 0 is `-1`. A frame scrolls only if the shifted overlap matches ≥ `match` and the unshifted frames match < `match`.
  - `step_starts(directions: np.ndarray, still: int = 3) -> np.ndarray` — `(m, 2) int64` rows `[frame, bit]`: a scrolling frame preceded by ≥ `still` non-scrolling frames, or whose direction differs from the previous scrolling frame.

- [ ] **Step 1: Write the failing tests**

```python
def test_scroll_reads_each_direction():           # random 144x160 index image, shifted 2 px
    for (dy, dx), bit in SHIFTS.items(): assert scroll(np.stack([img, shifted(img, dy, dx)]))[1] == bit
def test_still_and_partial_motion_are_not_scroll(): # identical frames; only a 40x40 sprite region moved
    assert scroll(...).tolist() == [-1, -1]
def test_step_starts_after_still_frames_or_turns(): # dirs = [-1]*4 + [2]*8 + [-1] + [2]*8 + [0]*8
    assert step_starts(dirs).tolist() == [[4, 2], [21, 0]]
```

- [ ] **Step 2: Run** `uv run pytest tests/test_label_rules.py -q` — expected FAIL (module missing).
- [ ] **Step 3: Implement** `SHIFTS`, `scroll`, `step_starts` in `poke/label_rules.py`; vectorise over the batch (compare `frames[1:]` to shifted `frames[:-1]` per shift).
- [ ] **Step 4: Run tests** — expected PASS.
- [ ] **Step 5: Measure G1 (walking) on true labels.** On the `clean` frames of the first 8 episodes in `data/synth/collection.json`, outside battles and text boxes: share of step starts with the direction held in `t-3 .. t-1`. Expected ≥ 0.98 (measured 0.987 by hand). Write `results/label_rules_synth.json` `{"walking_true": ...}`.
- [ ] **Step 6: Commit** `poke/label_rules.py tests/test_label_rules.py results/label_rules_synth.json` — "Read walking off the screen: a step scrolls 2 px a frame".

### Task 2: The walking rule

**Files:**
- Modify: `poke/label_rules.py`
- Test: `tests/test_label_rules.py`

**Interfaces:**
- Consumes: `step_starts` (Task 1).
- Produces:
  - `walking_agreement(masks: np.ndarray, starts: np.ndarray) -> float` — share of starts with their bit held in `t-3 .. t-1`.
  - `enforce_walking(masks: np.ndarray, starts: np.ndarray, held_p: np.ndarray, floor: float = 0.05) -> tuple[np.ndarray, np.ndarray]` — new masks and an `(n,) bool` of frames changed. For an unsatisfied start: hold the bit on `t-2 .. t-1` and clear the other direction bits there, unless `held_p[t-8:t+1, bit].max() < floor` (scripted walking).

- [ ] **Step 1: Failing tests:** `test_missing_step_press_is_inserted` (bit set on `t-2, t-1`, others cleared there, `changed` exactly those frames); `test_satisfied_start_is_left_alone`; `test_rule_never_removes_a_press` (a press with no step stays); `test_scripted_walk_gets_no_press` (`held_p` all 0 → unchanged).
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement.** **Step 4: Run** — PASS.
- [ ] **Step 5: Measure G2 (walking).** Decode v3 (`checkpoints/idm_v3/last.pt`) on 16 held-out synthetic episodes (`scripts.train_idm.split`, seed 0), enforce, and count inserted frames whose bit is held in the true labels. Expected ≥ 0.90. Add `"walking_inserted_precision"` and before/after `walking_agreement` to `results/label_rules_synth.json`.
- [ ] **Step 6: Commit** — "The walking rule: a step's direction is held before it starts".

### Task 3: The `▼` prompt and the text rule

**Files:**
- Modify: `poke/label_rules.py`
- Test: `tests/test_label_rules.py`

**Interfaces:**
- Produces:
  - `PROMPT_BOX = (slice(128, 136), slice(144, 152))` (tile row 16, column 18) and `PROMPT_GLYPH: np.ndarray` `(8, 8) bool`, True on the glyph's dark pixels — copied from the cartridge in Step 1.
  - `prompt_visible(frames: np.ndarray) -> np.ndarray` — `(n,) bool`: the box at `PROMPT_BOX` matches the glyph on ≥ 60 of 64 pixels (dark where `PROMPT_GLYPH`, the lightest colour elsewhere).
  - `text_change(frames: np.ndarray) -> np.ndarray` — `(n,) float32`, the share of the text area (rows 104–143, `PROMPT_BOX` excluded) that differs from the previous frame; 0 for frame 0. A per-frame number, so the whole video's worth fits in memory.
  - `answers(visible: np.ndarray, change: np.ndarray) -> np.ndarray` — `(m, 2)` rows `[first_visible, answered]`: a wait starts at a visible frame not already inside a wait, and is answered at the first later frame with `change > 0.02`. The blinking glyph alone changes < 1 % of the area, so a blink never ends a wait.
  - `text_agreement(masks, spans) -> float` and `enforce_text(masks, spans, onset_a: np.ndarray) -> tuple[np.ndarray, np.ndarray]` — needs an A onset (A held at `t`, not at `t-1`) inside `[first_visible, answered]`; if missing, adds a 5-frame A tap (`scripts.label_video.TAP_FRAMES`) at the `onset_a` argmax in the span.

- [ ] **Step 1: Copy the glyph from the cartridge.** From `states/cgb/bedroom.state`, walk to the PC and press A; tick until `wTileMap` (`poke.symbols`, 0xC3A0) holds 0xEE at row 16, column 18 (`hlcoord 18, 16`; if another tile id appears there while the text waits, use that and note it). Record the 8×8 block's dark pixels as `PROMPT_GLYPH`, and check that the blinking glyph changes < 1 % of the text area. If the ROM is absent, skip with a message.
- [ ] **Step 2: Failing tests:** `test_prompt_is_found_on_its_tile` (frame with the glyph pasted at `PROMPT_BOX`); `test_a_blinking_prompt_is_one_wait` (visible, invisible for 20 frames, visible again, then a 10 % text change → one span); `test_answer_needs_a_new_a_press` (A held from before the span → `enforce_text` adds a tap; an onset inside → unchanged); `test_text_tap_goes_where_the_head_peaks`.
- [ ] **Step 3: Run** — FAIL. **Step 4: Implement.** **Step 5: Run** — PASS.
- [ ] **Step 6: Measure G1 and G2 (text)** as in Tasks 1–2, on synthetic episodes with dialog: true-label `text_agreement` ≥ 0.97; inserted A taps with a true A onset within ± 8 frames ≥ 0.90. Add to `results/label_rules_synth.json`.
- [ ] **Step 7: Commit** — "The text rule: a ▼ answered means A was pressed".

### Task 4: Apply the rules to the whole video

**Files:**
- Create: `scripts/check_labels.py`
- Modify: `.github/workflows/ci.yml` (add `check_labels` to the `--help` loop)
- Test: `tests/test_check_labels.py`

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces:
  - `screen_events(video, chunk: int = 4096) -> tuple[np.ndarray, np.ndarray, np.ndarray]` — `(scroll (n,) int8, prompt (n,) bool, change (n,) float32)` over every video frame, read through `frame_ref` in chunks with a one-frame overlap; cached to `data/longplay/screen_events.npz`.
  - CLI: `--video`, `--labels`, `--name`, `--apply`. Writes `results/label_consistency_<name>.json`: steps found, `walking_agreement` before/after, prompt answers found, `text_agreement` before/after, frames changed. With `--apply`, writes `buttons_ruled (n,) u1` and `rule_changed (n,) bool` into the labels file.
  - Frames with `cut` or `state == 255` (all buttons) are dropped from both event lists.

- [ ] **Step 1: Failing test** `test_check_labels_end_to_end(tmp_path)`: a 60-frame fake video (`uframes`, `frame_ref`, `colours`) with one 8-frame left scroll from frame 20 and labels with no presses → after `--apply`, `buttons_ruled[18:20]` hold left, `rule_changed` exactly frames 18–19; a cut at frame 40 with a shifted-looking pair yields no step there.
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement.** **Step 4: Run** — PASS, and `uv run python -m scripts.check_labels --help`.
- [ ] **Step 5: Run on v3** (`nice -n 19`, measure peak memory first on `--limit 100000`; expected < 2 GB): `uv run python -m scripts.check_labels --name v3 --apply`. Record the whole-video agreement numbers — this is v3's G6 baseline.
- [ ] **Step 6: Commit** script, test, CI line, `results/label_consistency_v3.json` — "Check every step and every ▼ in the video against the labels".

### Task 5: Exports and repair use the ruled labels

**Files:**
- Modify: `scripts/export_video.py` (`label_arrays`, both exporters), `scripts/repair_labels.py` (`main`), `docs/longplay_labels.md`
- Test: `tests/test_export_video.py`

**Interfaces:**
- Produces:
  - `label_arrays(labels) -> tuple[held, held_p, verified, ruled]` — masks from `buttons_repaired` on verified frames, else `buttons_ruled`, else `buttons`; `ruled` = `rule_changed` or zeros.
  - PokeDreamer chunks and the SCHEMA-3 file gain `rule_events (N,) u1` (step starts + answers inside the step) and `rule_fixed (N,) u1` (frames the rules changed), beside `verified`.
  - `repair_labels` starts from `buttons_ruled` when present (`--buttons`, default `auto`).

- [ ] **Step 1: Failing tests:** `test_repaired_beats_ruled_beats_decoded` (three datasets differ on three frame ranges → `label_arrays` picks the right source per range); `test_export_carries_rule_counts` (extend the existing small-batch export test: the new arrays exist with the expected sums).
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement.** **Step 4: Run full suite** — PASS.
- [ ] **Step 5: Commit** — "Exports say which steps the screen checked".

### Task 6: Synthetic play aimed at the weak spots

**Files:**
- Modify: `poke/synth.py` (`HumanLike`, `play`), `scripts/synth_collect.py`
- Test: `tests/test_synth.py`

**Interfaces:**
- Produces:
  - `HumanLike(rng, start_rate=0.02, bump_a=0.0, text_hold=0.0)` — when the decision is A with a text box open, with probability `text_hold` hold A or B (50/50) for `_lognormal(rng, 30, 0.5, 10, 120)` frames instead of a tap. `text_hold=0` must leave the RNG stream unchanged.
  - `boost_encounters(env, symbols) -> None` — writes 255 to `wGrassRate` (0xD886).
  - `play(env, frames, rng, sample_every=20, policy=None, on_map_change=None)` — calls `on_map_change(env)` once before the first tick and on every frame whose `map_ids` differs from the previous frame's (each map load resets the rate from the map header).
  - `synth_collect` flags `--text-hold P` and `--encounters`, recorded in the manifest `options`.

- [ ] **Step 1: Failing tests:** `test_text_hold_holds_a_or_b_in_text_boxes` (fields with `dialog_opens` True, `text_hold=1.0` → runs of A or B ≥ 10 frames); `test_text_hold_off_keeps_the_old_stream`; `test_boost_encounters_writes_the_rate` (fake env recording `write_ram` calls); `test_play_calls_back_on_each_map_change` (fake env whose map id changes at frames 3 and 7 → callback at start, 3, 7).
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement.** **Step 4: Run** — PASS.
- [ ] **Step 5: Smoke** 2 episodes of each kind into the scratchpad (`--dry` folder), check: battle share ≥ 0.4 with `--encounters`; held A/B runs in text with `--text-hold 0.4`; peak RSS per worker ≤ 1.2 GB.
- [ ] **Step 6: Collect** under the watchdog, one batch at a time:
  `synth_collect --episodes 100 --first 30000 --frames 3000 --bump-a 0.6 --text-hold 0.4 --tag text --workers 4 --skip-existing` then
  `synth_collect --episodes 100 --first 31000 --frames 3000 --encounters --tag battle --workers 4 --skip-existing`.
- [ ] **Step 7: Commit** code, tests, `data`-free logs (`results/synth_text.log`, `results/synth_battle.log`) — "Synthetic play that holds A through text and fights more".

### Task 7: Train v4, label the video, check every gate

**Files:**
- Modify: `scripts/eval_idm.py` (add `--by-context`)
- Test: `tests/test_idm.py`

**Interfaces:**
- Produces: `context_table(truth, pred, matter, in_battle, dialog) -> dict[str, dict[str, list[int]]]` — per context (`overworld`, `text box`, `battle`) and button: `[caught, missed, wrongly pressed, correctly released]` on frames where the press could show; `eval_idm --by-context` writes it beside the closure numbers.

- [ ] **Step 1: Failing test** `test_context_table_counts_by_situation` (hand-built 6-frame arrays with known outcomes). **Step 2–4:** fail, implement, pass.
- [ ] **Step 5: Train** (watchdog, nice 19): `train_idm --init checkpoints/idm_v3/last.pt --steps 6000 --lr 1.5e-4 --out checkpoints/idm_v4`.
- [ ] **Step 6: Evaluate, one GPU job at a time:** `eval_idm --checkpoint checkpoints/idm_v4/last.pt --by-context` (G5, and the context table against v3's), then `label_video --checkpoint checkpoints/idm_v4/last.pt --out data/longplay/labels_v4.h5`, `check_labels --labels data/longplay/labels_v4.h5 --name v4 --apply` (G6), `sync_opening --labels data/longplay/labels_v4.h5 --buttons buttons_ruled` from power-on and from `states/cgb/bedroom.state` (G3), `repair_labels --labels data/longplay/labels_v4.h5 --frames 14000 --name poweron_v4` (G4).
- [ ] **Step 7: Decide.** Promote v4 (`labels_v4.h5` → `labels.h5`, v3's to `data/longplay_v3/`) only if G3, G5 and G6 pass; otherwise keep v3 with the rules applied (Task 4) and record why. Write the gate table to `results/longplay_v4_gates.json`.
- [ ] **Step 8: Commit** — the gate results, whichever way they went.

### Task 8: Export, document, ship

**Files:**
- Modify: `docs/longplay_labels.md`, `README.md` (results row 3b only if a number changed)

- [ ] **Step 1:** `export_video` on the promoted labels; check the export report counts (steps, `verified`, `rule_events`, `rule_fixed`).
- [ ] **Step 2:** Docs: what the two rules are and their measured soundness (G1/G2), the whole-video agreement before/after (G6), v3 vs v4 on G3–G5, what "verified" / "rule-checked" / unchecked mean for a user of the data.
- [ ] **Step 3:** `uv run pytest -q` passes; every relative link in README and docs resolves.
- [ ] **Step 4: Commit** — "Longplay labels checked across the whole game".
