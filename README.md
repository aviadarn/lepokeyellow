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

## Requirements

- Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/)
- **Your own legally acquired Pokémon Yellow ROM**, as `pokeyellow.gb`. No ROM,
  game asset or save state is included or distributed here, and `.gitignore`
  blocks all three.
- A graphical desktop for `capture_state` and `record_route`. A GPU is optional
  — see the measured numbers below.

## Run

```bash
uv sync
uv run pytest                                  # 59 tests, no ROM and no GPU needed

# Play to inside Oak's lab with an EMPTY party. Close the window to save.
uv run python -m scripts.capture_state pokeyellow.gb

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

Everything that does not need a ROM is tested and passing: ROM header
identification, the version RAM maps, the task predicates, route validation,
the baseline policies against a scripted fake emulator, the HDF5 round trip,
goal-frame selection, and the AUC against a brute-force pair count.

**Nothing that touches a real ROM has been executed.** No cartridge dump was
available while this was written, so the save state, the recorded route, the RAM
calibration, data collection, training and planning are all unrun. Expect to fix
things on the first pass. In particular `yellow_pikachu`'s default horizon of 40
is a guess at how many actions the walk plus the dialogue takes; `record_route`
reports the real number, and `calibrate_ram` is the first thing to run because
Yellow's `cur_map` is still an inference.

## Roadmap

1. Baseline table for `yellow_pikachu`, before any model exists.
2. Reproduce the pipeline end to end on Yellow; report the planner beside every
   constant-button policy.
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

## Licence

[MIT](LICENSE). Vendored and adapted portions are MIT from
[stmonty/lePokeRed](https://github.com/stmonty/lePokeRed) — see
[NOTICE](NOTICE) and [LICENSE.upstream](LICENSE.upstream) for the file-by-file
breakdown and the upstream commit.

Pokémon is a trademark of Nintendo / Creatures Inc. / GAME FREAK Inc. This
project is unaffiliated with them and distributes no game content.
