# Reading the buttons off a 7-hour longplay

Back to the [README](../README.md).

The second half of the PokeDreamer work
([pokedreamer_yellow.md](pokedreamer_yellow.md)) reads a human's presses off a
YouTube longplay of Yellow
(`aqr_YgqF2vM`, 6 h 53 min) with an inverse dynamics model, VPT-style, and
writes them in the same schema. The video is exactly 5× the Game Boy screen at
59.94 fps, one video frame per game frame, recorded through Gambatte's colour
correction; `scripts/video_ingest.py` undoes it and snaps every pixel to
Yellow's 42 colours (1,485,634 frames, 528,782 distinct).

`scripts/synth_collect.py` plays the emulator a frame at a time with
human-like timings and pushes every episode through the video's own path
(colour correction, 5× upscale, x264 at the longplay's CRF 28, decode, snap).
Labels per frame: the 8 held buttons, and on one frame in 20 whether flipping
each button would change the screen in the next 32 frames. 540 episodes, 1.8 M
frames: 320 from the start-state pool, 160 where a walk that goes nowhere is
followed by A (people, signs, PCs: 3 such episodes opened 10 text boxes
with A, 120 ordinary ones 19), 60 from power-on, half of them leaving the
intro untouched as the longplay's player did. `scripts/train_idm.py` trains
a patchified CNN + 4-layer transformer over 33 frames with four heads: held,
press onset, could-it-show, and edit cut (trained on spliced windows).

Two checks, because accuracy on held bits says little — on frames where a
press could show, predicting "nothing" is right 93 % of the time:

| | v2 | v3 |
|---|---|---|
| Synthetic replay, same tile path (12 held-out-map episodes × 300-frame segments; ceiling: true buttons one frame late, 0.842; floor: nothing pressed, 0.158) | 0.642 | 0.608 |
| Real video, frames in sync replaying from power-on | 923 | 1,816 |
| Real video, frames in sync replaying from the bedroom save state | 460 | 843 |
| Edit cuts found in 6.9 h (the cut head alone fired 4,294 times) | — | 330 |

The synthetic score cannot see A presses (they rarely change the path walked);
the video can, and v3 got past the bedroom PC where v2 missed the A. The cut
head fired on the game's own palette fades and on menus; `label_video` now
keeps a cut only where under half the pixels survive and no colour-for-colour
swap explains them.

`scripts/repair_labels.py` then lets the emulator correct the labels. It
replays them from power-on, and where the screen stops matching the video it
tries one small edit at a time — drop a press, move it, add an A / B / START
tap or an A / B hold (held A or B speeds up text, and the model never learned
that) — keeping the edit that plays in sync longest, with backtracking. From
power-on the labels are now verified frame by frame through the intro, the
title, OPTION, Oak's speech, both naming screens and the bedroom into the
PC's withdraw menu: to video frame 8,097, from v4's labels with 6 edits
(v3's needed 18 and stopped at 7,670; see below). Two things it cannot do: tell apart buttons
with the same effect (A or START on the title screen), and go past the
opening — verification needs an emulator state that matches the video, and
edit cuts and the random number generator end that.

## Checking the whole game, not just the opening

Past the opening nothing can be replayed, but two things the screen itself
shows still say a button was pressed (`poke/label_rules.py`,
`scripts/check_labels.py`, all 1,485,634 frames in 84 s):

- **A step.** The overworld scrolls 2 px on every other frame while the
  player walks, 16 frames a step, with the direction held just before it.
  True synthetic labels satisfy that on 99.0 % of 300 steps.
- **A `▼` answered.** The prompt (tile 0xEE, copied off the cartridge) waits
  for a new A or B press. B answers ordinary text: 11 of the 12 true-label
  waits with no A were answered with B. True labels satisfy it on 98.8 % of
  81 waits.

| | v3 | v4 |
|---|---|---|
| Steps found in the video, labels explaining them | 5,909, 95.9 % | 5,906, 96.2 % |
| ▼ answers found, labels explaining them | 4,875, 95.3 % | 4,875, 95.0 % |

Only the text rule changes labels. Where a wait has no press, it adds an A
where the onset head peaks: with the true press deleted from 42 held-out
synthetic waits, its A lands within 8 frames of it every time (and the true
press was A in 38, B in the others). On the video it added 227 A taps. The
walking rule stays a check: the steps the labels miss are forced movement
(spinner floors), where adding a press was always wrong.

v4 fine-tuned v3 on 200 more episodes aimed at its weak spots, half holding
A or B through text, half with a wild battle on every grass step. It replays
the video from power-on twice as far (3,081 frames against 1,816) and its
labels repair with a third of the edits, but it missed three gates narrowly
(bedroom replay 841 against 843, synthetic same-path 0.592 against the 0.60
bar, ▼ answers 95.0 % against 95.3 %) and is no better at A in text boxes
(4 of 12 against 5 of 12). So v3 with the text rule labels the unverified
frames, and v4's verified opening is kept: a replay in sync proves those
frames whichever model proposed them (`results/longplay_v4_gates.json`).

## Outputs and how far each step is checked

In `data/longplay/`: `labels.h5` (per frame: held / onset / could-it-show
probabilities, decoded, rule-checked (`buttons_ruled`) and repaired buttons,
`verified`, `rule_event`, `rule_changed`, confirmed cuts), `pokedreamer/`
(61,563 PokeDreamer steps, each with its 24 per-frame held masks and a
confidence) and `longplay_schema3.h5` (24,544 steps for `scripts/train.py`).

Every exported step says how far it is checked:

| field | meaning | PokeDreamer steps |
|---|---|---|
| `verified` | replayed in the emulator frame by frame | 289 |
| `rule_events` | steps and ▼ answers the screen shows inside it | > 0 on 10,234 (16.6 %) |
| `rule_fixed` | frames the text rule changed | 1,128 frames |

A step with events and no fixes is one the model got right on its own by the
screen's account; a step with no events is the model's reading, unchecked. RAM probes are
unknown (255 / −1): a video has no RAM. The frames come from someone else's
upload, so nothing derived from them is published.

## Reproduce

Each step is a script; every one takes `--help`. Run them one at a time: the
synthetic-play workers need about 1 GB each, and labelling or evaluating needs
about 6 GB of GPU memory.

```bash
# The video (YouTube format 298, 800x720 H.264) into data/video/, then:
uv run python -m scripts.video_ingest            # -> data/longplay/video.h5

# Synthetic play that looks like the video (data/synth/).
uv run python -m scripts.synth_collect --episodes 320 --frames 3000 --workers 4
uv run python -m scripts.synth_collect --episodes 160 --first 20000 --frames 3000 \
    --bump-a 0.6 --tag bump --workers 4
uv run python -m scripts.synth_collect --episodes 0 --intro 30 --frames 6000 --workers 4
uv run python -m scripts.synth_collect --episodes 0 --intro 30 --intro-first 1000 \
    --frames 6000 --intro-idle 1500 3600 --tag intro_idle --workers 4

# Train (v2 from scratch, v3 fine-tuned from it) and check by replay.
uv run python -m scripts.train_idm --steps 15000 --out checkpoints/idm_v2
uv run python -m scripts.train_idm --init checkpoints/idm_v2/last.pt --steps 6000 \
    --lr 1.5e-4 --out checkpoints/idm_v3
uv run python -m scripts.eval_idm --checkpoint checkpoints/idm_v3/last.pt

# Label the video, check it against the screen, repair the opening, export.
uv run python -m scripts.label_video --checkpoint checkpoints/idm_v3/last.pt
uv run python -m scripts.check_labels --name v3 --apply
uv run python -m scripts.repair_labels --frames 14000
uv run python -m scripts.sync_opening --buttons buttons_repaired
uv run python -m scripts.export_video

# v4 (not promoted): 200 more episodes, then fine-tune from v3.
uv run python -m scripts.synth_collect --episodes 100 --first 30000 --frames 3000 \
    --bump-a 0.6 --text-hold 0.4 --tag text --workers 4
uv run python -m scripts.synth_collect --episodes 100 --first 31000 --frames 3000 \
    --encounters --tag battle --workers 4
uv run python -m scripts.train_idm --init checkpoints/idm_v3/last.pt --steps 6000 \
    --lr 1.5e-4 --out checkpoints/idm_v4
```
