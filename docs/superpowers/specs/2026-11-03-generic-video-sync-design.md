# Generic emulator sync for the longplay — design

## Intent

The user wants the emulator, pressing the recovered buttons, to play the
longplay video (`aqr_YgqF2vM`, 6 h 53 min) **the same as the video**: every
piece between edit cuts replays identical to the footage under the strict
check, and sampled minutes from across the whole game prove it.

Why a new design: today's repair (`scripts/repair_labels.py`) is a list of
special cases. Each new kind of miss needs new code and a hand diagnosis.
The 2:32 miss (frame 9,116) is the example. The player stops beside a girl in
Pallet Town, turns right with a tap and presses A. The labels hold DOWN
about 5 frames too long, hold RIGHT for 157 frames and miss the A. Fixed by
hand, the emulator talks to her too. The repair could not find it, for three
reasons:
- every edit must gain MARGIN (30) frames alone;
- a release can move only 8 frames;
- chains stop at 3 edits.

The user wants a generic method, not a fix for this miss.

## Measured facts this rests on

- **The judge** (`scripts/sync_opening.in_sync`): a frame is in sync when
  ≥ 98 % of its pixels match and no 8×8 text patch differs by ≥ 6 px, with
  a 12-frame timing slack. Sync is lost after 20 frames in a row out of sync.
  - The check never fired on 1.26 M synthetic frames replayed with the true
    buttons (`results/text_check_synth.json`).
- **Verified today:** power-on to frame 9,116 (`results/repair_poweron_strict.json`),
  confirmed by a fresh replay (`results/sync_opening_strict.json`).
- **Emulator cost** (PyBoy 2.8.1, CGB, this Mac):

  | operation | cost |
  |---|---|
  | save state | 9.7 ms (211 KB) |
  | load state | 3 ms |
  | one frame, rendered | 0.066 ms (15 k fps) |
  | hash of WRAM | 0.1 ms |

  So the search replays candidates from snapshots. Forking every frame
  costs about 200 times as much as a frame.
- **Randomness:**
  - The state lives in `hRandomAdd` 0xFFD3 and `hRandomSub` 0xFFD4. Both
    change every frame: the VBlank handler calls `Random_`, which mixes in
    `rDIV`.
  - `rDIV` advances 18 per frame (mod 256).
  - PyBoy does not expose its timer object. Setting the timer phase means
    editing the save-state bytes.
- **Cuts:**
  - The cut detector confirms 330 cuts, median 0.8 min apart.
  - At least the first two are false: 1:53 and 2:09 fall inside the
    verified opening.
  - So the detector is a hint, not ground truth.
- **No map locator:** nothing in the repo finds which map and tile a video
  frame shows.

## Goals

1. **Generic button search.** Fix any mislabel without code for that case:
   - a missed press of any button;
   - a false press;
   - a hold too long or too short by any amount;
   - the wrong button;
   - presses closer together than one edit can prove.
2. **Randomness matching.** Where NPC moves, encounters or battle rolls
   differ, find random-number state that makes the emulator match the video,
   instead of stopping there.
3. **Cuts.** Tell real edit cuts from fades and menus. After each real cut,
   re-anchor from a save state rebuilt to match the screen.
4. **The test (the user's success measure).** Sample minutes from across the
   whole video, as checkpoints on one chained run from power-on (the user's
   choice). Each window must replay the same as the video. Report frames in
   sync per window.
5. **Every stop explained.** Give the frame and a reason:
   - buttons unexplained;
   - randomness unexplained;
   - edit cut;
   - hidden state (for example a Pokémon's unseen stats);
   - unknown.

## Non-goals

- Retraining or changing the IDM.
- Claiming the player's exact keys where several key sequences give the same
  screens (the IDM's likelihood picks among them).
- Publishing anything, or using a GPU or cloud.

## Anti-goals: these make it a failure even if it "works"

- **"Verified" that lies.**
  - A fix accepted without the strict judge.
  - Frames skipped unjudged (the snapshot bug fixed on 2026-11-03).
  - A number in a report that a fresh replay does not reproduce.
- **Hand-tuned per-case code.** For example a special edit for "turn then
  talk".
- **Slowing the user's Mac.**
- **Silently changing hidden game state.** Every random-number or state write
  is recorded and replayed by the fresh check.

## Constraints

- **The judge is fixed.** `in_sync` and its constants stay as they are. Any
  change must keep 0 false alarms on the synthetic set, and must still catch
  the wrong name, the wrong menu cursors and the K/RED case.
- **Verified means a fresh replay agrees.** Frames are marked verified only
  when a fresh replay from the anchor agrees: same buttons, same recorded
  state writes, a new emulator. Search bookkeeping alone never counts.
- **Mac safety.**
  - Jobs run at nice 19.
  - At most 4 worker processes, about 1 GB each.
  - One heavy job at a time.
  - The memory watchdog and screen sessions stay, as in
    `scratchpad/watchdog_strict.sh`.
- **Nothing derived from the ROM, a state or the video is committed.**
- **Existing interfaces stay:**
  - `labels.h5` keeps `buttons_repaired` and `verified`, written through
    `write_repair`, which clears any older reach.
  - `scripts/export_video.py` reads them unchanged.

## How it works — the parts decided

### Part 1: generic button search (replaces near/far/chain)

**Edit grammar.** Presses are intervals of one button held, with 8 buttons
and every one allowed. One edit is one of:

| edit | what it does |
|---|---|
| insert | add a press `[t, t+len)`, `len` ∈ {tap, 20, 60} |
| delete | remove a press |
| move start | shift a press's start by an offset from a geometric set in both directions, ±1 … ±256, clipped to the window |
| move end | the same, for its end (a 157-frame hold can become a tap) |
| split | remove the middle of a press, leaving two |
| change button | turn a press into the same interval on another button |

**Starting labels (my call).** Use v3's `buttons_ruled` with v3's `held_p`
and `onset_p` (`labels.h5`) everywhere, power-on included. Every window must
use these labels anyway, and the prior must come from the model that made
them. So the Part 1 test starts from scratch. It must re-find today's 22
fixes, v4's 6, and the intro's false A at 1,785.

**Window.** Edits touch `[lost − W, lost + 5)`.
- W starts at 300 frames.
- It doubles, up to 2,400, when a round finds nothing.
- A wrong press can show its effect much later: the false A at 1,785 shows
  about 1,500 frames on.

**Prior.** The prior is the change in log-likelihood of the edited frames
under the IDM:
- `held_p` per frame and button;
- `onset_p` for new presses.

It orders candidates and breaks ties. Nothing in it is specific to one
button.

**Search: a beam over edit sets.**
- The frontier holds up to B partial solutions (masks, reach).
- Each round expands every member by all single edits at its own loss of
  sync.
- It keeps the B furthest-reaching. Ties go to the higher prior, then to
  fewer edits.
- It stops when a member reaches `lost + HORIZON` or after D rounds.
- Then it commits the furthest member if it beats the start by MARGIN.
  Otherwise the point is stuck, and backtracking runs as today.
- So small gains add up across rounds. MARGIN applies to the whole committed
  set, never to one edit.

**Judging.** Every candidate replays from a snapshot at or before
`min(first edited frame, lost)`. Today's test for this stays.

**Cost: the judge comes first (measured by the reviewer).** Today's strict
run judged 1,438,821 frames in 81 min, about 3.4 ms per judged frame. That
is about 50 times the emulator alone, and the full grammar has about 10 k
candidates per beam member. So the first plan must:
- **Make the judge cheap.** Compare at the tracked offset first, and search
  neighbours only when that frame differs. Run the text check only where
  pure paper/ink patches differ. Target ≤ 0.5 ms per judged frame, with
  identical verdicts on the 9,116 verified frames and on the synthetic set.
- **Screen candidates.** Try them in prior order, with a cap per member per
  round (about 400, tuned), plus the existing early abort after RUN bad
  frames.
- **Targets (my call):** a stuck point takes ≤ 5 min (median) and ≤ 30 min
  (90th percentile). The Part 1 run, from power-on to 3:00, finishes in one
  night (≤ 8 h).

**Success.**
- From power-on, unattended, with no hand edits, it reaches at least 9,116.
  That means it finds today's fixes or equivalent ones on its own.
- It gets past 2:32: the whole conversation with the girl, plus HORIZON
  frames after it, passes the strict judge.
- A fresh replay confirms the result.

**The girl's text-speed difference comes first (my call).** It is
unexplained. Holding A did not change it, and holding B was never tried.
Find its cause before building the search.
- If it is a button, the search must find it.
- If it is not, the classifier must name it, and Part 1's test becomes "an
  explained stop at the girl".

### Part 2: random-number matching

**When.** At a stuck point the classifier (Part 4) calls random, for
example when the UI is identical and only sprite positions or battle numbers
differ.

**Search.**
- Search `(hRandomAdd, hRandomSub)` at a frame shortly before the
  divergence: 65,536 values.
- Test each over a short horizon, about 30 frames.
- With today's judge that is about 100 ms per value: about 110 min, or
  about 27 min on 4 workers.
- Screen first with a cheap compare of only the pixels where the event
  shows. Target ≤ 10 min per resync.
- Then keep going with the button search.

**Spike first.** Before building it out, measure how long one resync stays
in sync. The answer picks the build:
- If it holds for minutes, the two emulators' timing agrees once the bytes
  match: build per-event resync.
- If it breaks within seconds, the timer phase must match too: also search
  the timer counter by editing PyBoy's save-state bytes.

**Recording.** A resync is a state write, not a button. Each one (frame,
values) goes in the repair report and in the `labels.h5` attrs. The fresh
replay applies the same writes.

**Verified, by kind (my call).** `verified` stays as it is: buttons only,
from an exact anchor, with no state writes. Two new datasets sit beside it:
- `verified_resync`: frames that match given recorded random-number writes;
- `verified_rebuilt`: frames after a start state rebuilt with guessed values.

The export treats the buttons of all three as checked, so their confidence
is 1. It records which kind applies per step (`verified_mode`), so nobody
mistakes the three for one another.

**Success.** Passes the opening's first random events: Pallet NPC moves, the
first wild encounter, the rival battle at about 4:30.

### Part 3: cuts — carry the run's own state across them (the user chose A)

The user chose one chained run through the whole video over independent
windows. So the emulator plays from power-on and carries its own game state
across every cut, which needs the fewest guesses.

**A real cut** is a stuck point that neither buttons nor randomness explain,
where the screen jumps: a map change with no warp or fade, or the party,
level or money changing. The detector's cut list is a hint only. The
reviewer checked the confirmed "cuts" at 3:05, 3:07 and 6:37: they are
palette flashes and battle transitions, not edits.

**Carry and patch.** At a real cut:
1. Take the run's own emulator state from just before the cut.
2. Patch in what the first clean frame after it shows changed:
   - map, position and facing, by teleporting through the game's warp
     routine (`poke/warp.py`);
   - party species, levels, HP and moves where shown;
   - money, items, badges and options where shown.
3. Everything the screen never shows stays as carried. For example, items
   bought or used during the cut-out grinding.
4. Every patched value is recorded: shown or guessed, old and new value.

When a later stop traces back to a guess, the patch can be revisited.

**New tool: a map locator.** It finds which map and tile a frame shows. The
builder picks the method. It must place ≥ 95 % of overworld frames correctly
on a test set from Part A's exact data (`data/pokedreamer_yellow/`).

**The run takes days and must survive it.** On the user's Mac a full run
means days of background compute, through sleep and restarts. So:
- progress is checkpointed on disk (labels, edits, state writes, the
  emulator state at the frontier);
- a restart resumes from the last checkpoint;
- a kill loses at most one stuck point's search.

### Part 4: the sampled-window test, and stop reasons

**Windows are checkpoints on the chained run.**
- Twenty one-minute windows across the video, with a fixed seed.
- Stratified by situation: overworld, text, menu, battle.
- Each window is scored on the chained run as it passes, through a fresh
  replay of its span from the run's recorded state at the window's start.
- Windows the run has not reached yet show as pending. The report updates as
  the run goes.

**Stop classifier.** It reads the frames where sync is lost:
- **Random:** UI identical, only sprites or battle numbers differ.
- **Cut:** a jump with no transition.
- **Hidden state:** battle damage differs with the random numbers matched.
- **Buttons:** the default.
- **Unknown:** none of the above fits.

**Report.**
- `results/sync_windows.json`.
- A local-only page with side-by-side clips. The video frames are not
  published.

**Target.** Every window 100 % in sync, except stops classified cut or
hidden state. Those are reported honestly, never hidden.

## Order

1. Part 1, with the Part 2 spike and the girl diagnosis: the first plan. The
   run starts from power-on.
2. Part 2, as built after the spike. The run continues through the opening's
   random events.
3. Part 3: carry-and-patch at real cuts, and the map locator. The run
   continues through the video, resumable.
4. Part 4's report from the start, filling in as the run passes each window.

Each step ends with the fresh-replay check, and with the earlier results
reproduced.

## Left to the builder

- Data structures and file layout.
- The exact offsets and lengths in the edit grammar.
- B and D. Start at B = 4 and D = 12, and measure.
- Parallelism inside the 4-worker limit.
- The map-locating method.
- How snapshots are kept.
- The classifier's thresholds. Tune them on frames already seen: the 2:32
  girl (buttons), the title Pikachu (no-input art), the edit cuts the
  detector lists.
