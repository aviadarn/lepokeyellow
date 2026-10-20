# lepokeyellow

A JEPA world model and latent-space planner for **Pokémon Yellow**, ported and
extended from [stmonty/lePokeRed](https://github.com/stmonty/lePokeRed)
([blog post](https://nostalgia.dev/posts/teaching-a-world-model-to-play-pokemon/)),
which builds on
[LeWorldModel](https://arxiv.org/abs/2603.19312) and
[LeJEPA](https://arxiv.org/abs/2511.08544).

The model is unchanged: a 12.5M-parameter ViT encoder to a 192-dim embedding, a
button-conditioned causal predictor, MSE on the next embedding plus
`0.1 · SIGReg` to stop the representation collapsing. No rewards, no labels.
Planning is categorical CEM over button sequences, scored by distance to a goal
embedding.

What changed is everything around it, because **Yellow has no three-starter
choice.** Oak's lab holds a single Poké Ball, the rival takes it, and Oak then
hands you Pikachu through dialogue — nothing you do changes which Pokémon you
get. Upstream encodes its task by naming episodes `charmander` / `squirtle` /
`bulbasaur` and scanning the party count, in the collector, the planner and the
evaluator. None of that ports.

## Why the port needed a different task

Red's start state has the player already facing the middle ball, and upstream's
own training data ships the solution: `TAKE = ["a"] * 12`. The blog post reports
52/100 planner successes against controls of uniform random (0/100) and an
untrained predictor (1/100) — and then notes in passing that pressing A
repeatedly already works from that state. That row is not in its table.

So this repo makes tasks explicit, and ships the table:

| task | games | goal (read from RAM) | solvable by one repeated button? |
|---|---|---|---|
| `red_starter` | red | party count ≥ 1 | **yes** — `always a`. Kept only so Red numbers stay comparable. |
| `yellow_pikachu` | yellow | party count ≥ 1 | no — the desk must be walked to before any dialogue starts |
| `leave_room` | red, yellow | current map ≠ starting map | no — pure navigation |

`scripts/eval_baselines.py` runs all seven constant-button policies plus uniform
random and prints a warning if any constant button solves the task. Run it
before believing a planner number.

## What is new here

| file | why |
|---|---|
| `poke/games.py` | Per-version WRAM map, selected from the ROM header title. Yellow moved `wGBC` into HRAM, so every WRAM symbol above `$CF1A` sits one byte lower than Red/Blue: `party_count` is `0xD163` in Red and `0xD162` in Yellow. A wrong address does not raise — it returns the neighbouring byte, so Red-addressed code reads a Yellow *species index* and calls it a count. Addresses carry a `confidence` field; Yellow's `cur_map` is marked `inferred` because it comes from the shift rule, not from a disassembly. |
| `poke/tasks.py` | A task is a goal predicate, a horizon and a start state. The planner and evaluator take a task name, so Yellow needs no forked copies. |
| `poke/baselines.py` | Every constant-button policy plus uniform random, run in the emulator. The missing row. |
| `poke/routes.py`, `scripts/record_route.py` | Yellow's path across the lab depends on where the save state was made, so the demonstration cannot be written blind. Record it once by hand in the env's own action space (one keypress = one action), stored with the digest of the state it was recorded against. `collect.py` refuses a route whose digest does not match, because replaying a route from the wrong state produces confidently mislabelled episodes. |
| `scripts/calibrate_ram.py` | Confirms `party_count` on a real ROM by diffing all of WRAM across the demonstration and reporting every byte that went 0→1 once and never above 6. Needed because one Yellow address is inferred. |
| `poke/goals.py` | Start and goal frames come from a per-step goal flag written by the collector, not from starter names and not from `party_counts` — `leave_room` succeeds without the party changing at all. |
| `scripts/sweep_seeds.py` | The post's 100-seed repeatability test, keeping CEM's **predicted** cost beside the **actual** emulator outcome, and reporting the AUC between them. 0.5 means the planner cannot tell its good plans from its bad ones, i.e. CEM is exploiting model error rather than using model knowledge. |

Two fixes carried into the vendored code:

- **Device selection.** Upstream picks `cuda` or `cpu`, so training on an Apple
  box silently lands on CPU. Now resolves MPS too.
- **`probe.py` labels.** Labelled on `party_counts > 0`, which is constant for
  `leave_room`; now uses the goal flag.
- **`PokeDataset` error message.** An episode set with no window long enough
  raised numpy's "need at least one array to concatenate", which names neither
  the file nor the knob. `finetune_rollout --max-horizon` is the usual cause.
- **`train.py` flags.** `DefaultConfig` was edit-the-source-only; its fields
  are now CLI flags, along with `--checkpoint-dir` and `--device`.
- **`plan_cem.py` needs no ROM.** Planning never touches the emulator, and the
  dataset records its own game and task, so `--rom` is optional and only
  cross-checks.

One upstream behaviour deliberately dropped: `eval_plan.py --runs N`. The
emulator is deterministic from a save state and the plan is fixed, so N runs are
N identical episodes. Varying anything means varying the planner seed, which is
what `sweep_seeds.py` does.

## No hand-made save state

Upstream needs you to play the game: *"Play until you can choose a starter.
Stand below the middle Poke Ball... Close the window to save."* Every result
then rests on an artifact nobody else can reproduce, which is why routes have
to carry the digest of the state they were recorded against.

None of that is necessary. Yellow's entire intro -- Oak's speech, both name
entries, the walk-on -- is cleared by pressing A, and from the bedroom a seeded
random walk finds the way out. `scripts/bootstrap.py` presses A until the game
hands over control, searches for a target, truncates at first arrival, and
**replays the route from the start to check it lands in the same place** before
saving. Same seed and the same dump give the same file on any machine.

| target | from a bare ROM | result |
|---|---|---|
| `bedroom` | 74 A presses, **1.4 s** | controllable, map 38, party 0 |
| `oaks_lab` | + 765 search steps, **17 s** | map 40 (2 of 6 seeds found it) |
| `pikachu` | + 1,485 search steps, **25 s** | map 40, **party 1** (1 of 4 seeds) |

```bash
uv run python -m scripts.bootstrap --rom pokeyellow.gbc --to pikachu
```

Two bugs this flushed out, both recorded in the code:

- **B is backspace on the name-entry screen.** Pressing it speculatively to
  clear a text box bounces between the name menu and the letter grid forever;
  the first version burned 5,104 actions that way. Only A is ever committed,
  and any experiment happens between a snapshot and a restore.
- **The cutscene moves the player.** A probe that calls "position changed"
  control reports control that does not exist during the walk-on. The probe now
  runs a no-op of identical timing first, and requires the move to be
  reversible -- no cutscene honours that.

Snapshot fidelity was checked directly rather than assumed: position, screen
and WRAM all come back identical, and replay from a restored state is
deterministic.

## Addresses, confirmed on the cartridge

Yellow moved `wGBC` into HRAM, so WRAM above `$CF1A` sits one byte below
Red/Blue. The shift rule predicted these; running the ROM confirmed them.

| Yellow | Red | symbol | how it was confirmed |
|---|---|---|---|
| `0xD162` | `0xD163` | `party_count` | watched go 0 -> 1 when Pikachu arrives |
| `0xD163` | `0xD164` | `party_species` | documented |
| `0xD35D` | `0xD35E` | `cur_map` | reads 38 on a new game; 38 -> 37 -> 0 -> 40 along the route |
| `0xD360` | `0xD361` | `wYCoord` | 6 <-> 7 on down/up |
| `0xD361` | `0xD362` | `wXCoord` | 1 <-> 4 on left/right |
| `0xD056` | `0xD057` | `wIsInBattle` | **still inferred** -- reads 0, which a wrong address also reads |

Confirmed against ROM sha1 `cc7d0326...` (Yellow (UE) [C][!]), recorded in
`games.CONFIRMED_ON` so a later disagreement points at a different dump rather
than at the table.

**Yellow is a dual-mode cart** (CGB flag `0x80`), so PyBoy boots it in colour:
R, G and B differ. Every writer takes channel 0 and the encoder is
`Conv2d(1, ...)`, so colour would quietly feed the model an arbitrary channel
instead of luminance. `PokeEnv` forces DMG, giving the 4-shade grey screen the
blog post trained on. Upstream never hit this -- a `.gb` Red ROM has no colour
mode to pick.

## The task had to be redefined, and the table says why

The obvious framing -- start inside Oak's lab, acquire a Pokemon -- is not a
hard task, it is an impossible one. The lab is enterable long before the story
hands anything over; Oak has to stop the player north of Pallet first.
Bootstrapping to map 40 and running the baselines there:

| policy | hits |
|---|---|
| all seven constant buttons | 0/1 each |
| uniform random | 0/50 |

That reads like a brutally hard task and is really a start state the goal
cannot be reached from. A table of zeros is a broken task, not a good result.

Redefined from a new game, with a horizon matched to what random actually
needs (525-2,600 actions across seeds):

| policy | hits | rate |
|---|---|---|
| always noop / up / down / left / right / a / b | 0/1 each | 0.0% |
| **uniform random** | **1/50** | **2.0%** |

Reachable, but not by accident. That 2.0% is the floor any planner has to beat.

## Requirements

- Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/)
- **Your own legally acquired Pokémon Yellow ROM.** No ROM, game asset or save
  state is included or distributed here, and `.gitignore` blocks all three.
  Either the `.gb` or the GBC-enhanced `.gbc` dump works; the env forces DMG
  rendering either way.
- No graphical desktop and no GPU required. `bootstrap` is headless, and the
  measured numbers below are all from a laptop. `capture_state` and
  `record_route` still exist for hand-made states, but nothing needs them.

## Run

```bash
uv sync
uv run pytest                                  # 67 tests, no ROM and no GPU needed

# Bare ROM -> a verified, reproducible save state. No playing required.
uv run python -m scripts.bootstrap --rom pokeyellow.gbc --to bedroom
uv run python -m scripts.bootstrap --rom pokeyellow.gbc --to pikachu

# Sanity check: detected game, RAM readings, and whether the task is already solved.
uv run python -m scripts.smoke_test

# Demonstrate the route once, by hand. One keypress = one action.
#   w a s d = up left down right   j = A   k = B   . = noop   u = undo   q = save
uv run python -m scripts.record_route

# Confirm the RAM address against this ROM rather than trusting the table.
uv run python -m scripts.calibrate_ram

# The baseline table. Do this before training anything.
uv run python -m scripts.eval_baselines

# Collect, train, probe, fine-tune on longer rollouts.
uv run python -m scripts.collect --random 800 --random-steps 48 --noisy 200
uv run python -m scripts.train
uv run python -m scripts.probe checkpoints/best.pt
uv run python -m scripts.finetune_rollout checkpoints/best.pt

# Plan, execute, then repeat over 100 planner seeds with cost/outcome calibration.
uv run python -m scripts.plan_cem checkpoints/rollout/best.pt --out plans/yellow.json
uv run python -m scripts.eval_plan --plan plans/yellow.json --show
uv run python -m scripts.sweep_seeds checkpoints/rollout/best.pt --seeds 100
uv run python -m scripts.sweep_seeds checkpoints/rollout/best.pt --seeds 100 --untrained
```

## Cost

Measured on an Apple M5 (32 GB) with torch 2.13 on MPS, on this architecture at
`batch_size=16, num_frames=4`. No cloud GPU is needed for any of it.

| step | measured |
|---|---|
| parameters | 12,515,136 (encoder 5.41M, predictor 6.96M, action encoder 0.15M) |
| training throughput | 403 frames/s, flat from batch 16 to 64 — compute-bound |
| default 10-epoch run | **64 min** |
| encoder share of a training step | 146 of 155 ms = **94 %** |
| SIGReg share | 2.4 ms = **1.5 %** |
| inference | 1,758 frames/s |
| CEM, 512 plans, horizon 14 | 173 ms per round |
| 100 planner seeds, 15 rounds each | **~4 min** of GPU time |
| same at horizon 112 | ~20 min |

CEM cost is **linear** in the horizon, so testing the post's unverified claim
that difficulty grows exponentially with plan length is cheap. Swapping
`nn.MultiheadAttention` for SDPA is only a 1.17× win here (53.7 vs 62.8 ms per
12 blocks), so the encoder is genuinely the cost. The lever that works: the
4-frame windows overlap, so each frame is re-encoded about four times. That
cannot be cached while the encoder is still learning, but `finetune_rollout.py`
freezes it — caching embeddings there is roughly 4× off the expensive phase.

## Status

**Run on the real cartridge:** ROM identification, DMG forcing, all six RAM
addresses but `wIsInBattle`, the bootstrap to all three targets with replay
verification, the map graph, and the baseline table for both the broken lab
start and the working new-game start.

**Run against synthetic data only:** train, probe, rollout fine-tuning and CEM
planning all work end to end on a dataset in the collector's h5 format, on MPS.
They have not yet been run on frames from the game, because no dataset has been
collected yet.

**Not built yet:** the MPC play loop. Everything so far plans once, open-loop,
to one goal. Playing needs replanning against the real screen, a chain of
subgoals, data along the route, and a scripted policy for battles — see the
roadmap.

67 tests, none needing a ROM or a GPU.

## Roadmap

1. ~~Baseline table for `yellow_pikachu`, before any model exists.~~ Done:
   constant buttons 0, random 1/50.
2. Collect a dataset around the bootstrap states and train on real frames;
   report the planner beside every constant-button policy.
3. Rollout error as a curve, `MSE(k)` for `k = 1..H`, before and after
   fine-tuning. The post gives one number (step 12: 0.4224 → 0.3045) and says
   the first step got slightly worse without quantifying it. Cheap:
   `finetune_rollout` already prints per-horizon validation MSE, so this is a
   plot and a before/after column, not new machinery.
4. Success against horizon on `leave_room` from start states at increasing
   distance from a door — the post's exponential-difficulty claim, measured.
5. Does the planner know when it is wrong? `sweep_seeds` already records it;
   if the AUC is near 0.5, try an ensemble disagreement penalty.
6. Cross-version transfer: train on Yellow, score the frozen encoder and
   planner on Red's `leave_room` with no retraining. Same engine, different
   sprites and palette — does the latent space encode game structure or pixel
   surface? Needs a second legal ROM.
7. **Playing, properly.** Replace one open-loop plan with MPC: plan H, execute
   1–3, re-encode the real screen, re-plan. Follow a chain of subgoal images
   from the bootstrap states, hand control to a scripted policy while
   `wIsInBattle` is set (which needs confirming first), and report how far
   along the chain it gets against random, constant-A, and open-loop CEM.

## Licence

[MIT](LICENSE). Vendored and adapted portions are MIT from
[stmonty/lePokeRed](https://github.com/stmonty/lePokeRed) — see
[NOTICE](NOTICE) and [LICENSE.upstream](LICENSE.upstream) for the file-by-file
breakdown and the upstream commit.

Pokémon is a trademark of Nintendo / Creatures Inc. / GAME FREAK Inc. This
project is unaffiliated with them and distributes no game content.
