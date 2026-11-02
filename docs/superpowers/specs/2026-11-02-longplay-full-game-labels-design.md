# Longplay labels for the whole game — design

**Goal.** Make the button labels of the 6 h 53 min longplay (`aqr_YgqF2vM`,
1,485,634 frames, new game to credits) accurate across the whole video, not only
in the 7,670-frame opening that emulator repair can verify, and say per step how
far each label is checked.

**Why this route.** The video is the only start-to-end playthrough we have
(the emulator dataset has 0 badges on all 500k steps). Its labels exist for every
frame but are the model's guesses past frame 7,670. Emulator replay cannot follow
the video past the opening (edit cuts, RNG), so checks for the rest must come from
what the screen itself proves.

## Measured facts this design rests on (2026-11-02)

- **Walking is visible exactly.** On synthetic overworld frames the background
  scrolls 2 px per frame while the player walks: shift `(0,-2)` = left,
  `(0,+2)` = right, `(+2,0)` = down, `(-2,0)` = up (shift `(dy,dx)` such that
  `new[y,x] == old[y+dy, x+dx]`). A step is 8 scrolling frames.
- **A step's direction is held just before it starts.** For steps that begin
  after ≥ 3 still frames, the true labels hold the matching direction within
  frames `t-3 .. t-1` of the first scrolling frame `t` in **98.7 %** of 155 cases
  (`t-1`: 97.4 %, `t-2`: 98.1 %).
- **Where v3 is weak** (held-out synthetic, frames where a press could show):
  A in a text box caught 3 of 7, A in battle 7 of 10, `down` while walking 60 %,
  `up` 80 %; directions pressed wrongly on 3–8 % of walking frames.
- **Gen 1 text.** Holding A or B prints text faster; B does not advance a page;
  the `▼` prompt waits for A (README Part 1, "B is not an answer either").

## Decisions

1. **Game-rule decoding, applied to the whole video** (new `poke/label_rules.py`):
   - *Walking rule:* every step start seen in the scroll must have its direction
     held in `t-3 .. t-1`; if the labels do not, hold it on `t-2 .. t-1` and
     clear the other directions there. Never remove presses (a press into a
     wall scrolls nothing and is real).
   - *Text rule:* every `▼` prompt that is answered (the prompt disappears and
     the box changes) must have an A onset between the prompt appearing and the
     answer; if not, add a 5-frame A tap where the onset head peaks in that span.
   - Both rules are checked first on **true** synthetic labels (they must hold
     ≥ 98 % / ≥ 97 %), then on predicted ones (inserted presses must agree with
     truth ≥ 90 %).
2. **A v4 model** fine-tuned from v3 on 200 new synthetic episodes aimed at the
   weak spots: 100 with A/B *held* through text plus bump-A, 100 with frequent
   wild battles (`wGrassRate` set to 255 at collection time only).
3. **Per-step label quality in every export:** `verified` (emulator replay),
   plus two counts per step — `rule_events` (step starts and `▼` answers the
   screen shows inside it: how much there was to check) and `rule_fixed`
   (frames the rules had to change). A step with events and no fixes is one the
   model got right on its own; a step with no events is unchecked.
4. **One heavy job at a time on the Mac** (user working): nice 19, ≤ 4
   synthetic workers (~1 GB each), GPU jobs (~6 GB) never in parallel, the
   pause/resume watchdog from `scratchpad/watchdog_v3.sh` pattern.

## Not in scope

Matching the emulator to the video past the opening (map localisation,
re-anchored replays); badges/story flags in the emulator dataset; menu-cursor
rules; publishing anything derived from the video.

## Success gates

| | gate |
|---|---|
| G1 | true synthetic labels satisfy the walking rule ≥ 98 % and the text rule ≥ 97 % |
| G2 | presses the rules insert into v3's synthetic predictions agree with truth ≥ 90 % |
| G3 | v4 + rules on the video: power-on replay > 1,816 frames, bedroom replay > 843 |
| G4 | reported, not required: emulator repair from power-on needs fewer edits per 1,000 verified frames than v3 (18 per 7,670) |
| G5 | synthetic replay closure, same path, ≥ 0.60 (v3: 0.608) |
| G6 | whole-video report: share of video step starts and `▼` answers the model's own labels explain, v4 ≥ v3 |
