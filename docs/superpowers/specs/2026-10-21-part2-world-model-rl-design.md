# Part 2 with a world model + RL policy: design

Date: 2026-10-21. Status: draft for review.

## Intent

Play Pokémon Yellow through the end of Bulbapedia's walkthrough Part 2, entering
Viridian Forest, with a **learned policy that sits on the world model** doing
the navigation and the story interactions. The run must survive the critique
the Jev Plays Pokémon thread raised: a harness that reads RAM, computes paths
and leaves the model to flip a coin. So the deliverable is the run plus an
honest account of which parts were learned.

Today the repo plays a new game to Route 1 (commit `0b4d2df`: 4/4 Part 1
milestones on 2 of 3 seeds). It does that with the JEPA encoder, tile probes and
a pursuit/CEM controller that steers along **demo waypoints** with arrival read
from RAM. In run `play_milestones_s1` the planner chose 134 of 442 actions (30%).
The rest came from scripted controllers. There is no RL in the repo.

Prior art never did the parcel errand. PWhiddy's agent and Pleines et al.
(arXiv 2502.19920) both start after the parcel is delivered, because "the first
two objectives require backtracking".

## The play-time contract

The user decided this. It is the rule every reported run must obey.

**At training time anything goes:** RAM-read rewards, save-state resets,
oracle routes, archives of states. **At play time:**

| Piece | At play time |
|---|---|
| Navigation | **Learned.** Pixels → frozen world-model encoder → policy → button. No demo waypoints, no coordinate, map-id or wall reads, no BFS distances. |
| Talking: choosing to press A facing Oak, the clerk, a sign | **Learned.** It is how the story advances. |
| Current objective | **Read from flags** (user approved). The policy is told which objective it is on. |
| Advancing an already-open text box, and story menus | Scripted handler, declared. It never makes a story choice except declining (no nickname, no purchases). |
| Battles: Part 2 wild fights, the old-man catch demo, the rival in Part 1 | Scripted handler, declared. |
| Emulator save/restore | **Banned.** One continuous run, no rollback. The existing `has_control` probe snapshots and restores, so play time needs a rollback-free replacement. |
| Reward | Training only. Never a policy input. |

Every action is logged with the controller that chose it, as `play.py`
already does with `actions_by_controller`. Turns where nobody has control and no
text box is open get their own label, `no_control`. These are walking
cutscenes and the swallowed first input after a warp. They count as neither
learned nor handler actions.

The battle handler fights: it mashes A, which selects Thundershock, as the
spike did. A blackout is not special-cased. The run continues from wherever the
game respawns the player, and the cost shows up in the action count.

## Win condition and milestones

Measured on the cartridge (spike, 2026-10-21). Addresses are Yellow's; Yellow WRAM
is Red minus one above `$CF1A`.

| # | Objective id (policy input) | Done when | Where it flips |
|---|---|---|---|
| A1 | `get_parcel` | `wEventFlags` bit 57 `GOT_OAKS_PARCEL` = `0xD74D` bit 1 | Entering Viridian Mart, (42, 5, 2) |
| A2 | `deliver_parcel` | bit 56 `OAK_GOT_PARCEL` = `0xD74D` bit 0, and bit 37 `GOT_POKEDEX` = `0xD74A` bit 5 | Talking to Oak, (40, 3, 5) |
| A3 | `enter_forest` | `cur_map` (`0xD35D`) == 51, Viridian Forest | Through the gate building, map 50. **Map 51 confirmed** on the cartridge (2026-10-21, `results/oracle_forest.json`): 5 actions north from the gate state lands at (51, 47, 16). |

**The objective id comes from flags only** (and the bag for the parcel). It
never comes from map id or position, so the policy input never reads location.
"Done" checks may read the map (A3), because they judge the run and do not
steer it.

Supporting facts, all confirmed unless marked:

- `wEventFlags` base is `0xD746`. `wNumBagItems` is `0xD31C`, items follow as
  (id, qty) pairs, and Oak's Parcel is `0x46`.
- **Bit 57 never clears on delivery.** "Carrying the parcel" = parcel in bag.
  "Delivered" = bit 56.
- **Getting the Pokédex arms the optional Route 22 rival fight** (bits 1312 and
  1319 at `0xD7EA`).
- The Viridian old man gate is a **trigger tile (1, 9, 19)** that pushes the
  player down until the Pokédex. Measured: with the parcel and no Pokédex, the
  search covered all 498 reachable Viridian tiles and found no way to Route 2.
  After the Pokédex it got through.
- After the Pokédex, crossing (1, 9, 19) forces the catch demo, which sets
  bits 45 and 47.
- **The first input after any warp is swallowed by the fade-in.** The state does
  not change, so dedup or "no effect" logic reads every door as a dead end. One
  no-op restores control.
- Maps: Pallet 0, Viridian 1, Route 1 12, Route 2 13, Oak's Lab 40, Pokémon
  Center 41, Mart 42, forest south gate 50, forest 51.
- Route 1 has one-way ledges going south. Wild Pidgey/Rattata Lv2-7 on Route 1
  and Route 2. Blacking out before visiting a Pokémon Center respawns at home.
- One 60-frame action sometimes walks two tiles.

**Stage B extends the chain back to a new game.** Same mechanism, earlier
objectives, still flags only:

- `trigger_oak` while bit 39 `OAK_APPEARED_IN_PALLET` is 0.
- `get_starter` while bit 39 = 1 and bit 35 `BATTLED_RIVAL_IN_OAKS_LAB` = 0.
  The handover needs the right tile and facing, then A. The rival fight
  that follows is the battle handler's.
- **Leaving Pallet is folded into `get_parcel`**, which starts once bit 35 = 1
  and bit 57 = 0.

The user's win condition is the whole run through Part 2, so Stage B is part of
done.

## Architecture: how the world model and the RL policy mesh

My calls, open to change:

1. **Representation.** The existing JEPA encoder (192-d per frame, 12.5M-param
   model) is retrained on a dataset that covers the Part 2 maps, then frozen.
   The policy sees the embeddings of the last 4 frames.
2. **Policy.** A small actor-critic trained with PPO.
   - Input: 4 frame embeddings, the objective one-hot and the previous action.
   - Body: an MLP into a GRU for memory.
   - Output: logits over the 7 existing actions (noop, 4 directions, A, B), plus a value head.
   - Well under 1M parameters.
3. **Exploration bonus from the world model** (training only), in the
   Plan2Explore style. The intrinsic reward is latent prediction error, or
   disagreement across a small predictor ensemble, on transitions where the
   policy had control. It is never computed during handler steps, because text
   boxes and the Pikachu-follow sprite are noisy-TV sources.
4. **Imagination training** (Dreamer-style rollouts in the predictor) is **out of
   the first plan.** It is a follow-up experiment arm, only after 1-3 work.
   Reason: this repo measured that minimising latent goal distance is Goodharted
   (door harvesting, 15 round trips), and that multi-step rollouts drift.

The world model has to earn its place. A pixels-only PPO baseline with the same
reward, handlers and budget is mandatory (see Evaluation).

## Training-time machinery

- **Oracle.** Port the spike's BFS (scratchpad `flag_spike/spike.py`, including
  the warp no-op fix) into the repo as a script. It provides milestone save
  states (parcel, lab with parcel, Pokédex, forest gate) and demonstration
  frames covering maps 12, 1, 42, 0, 40, 13 and 50. Measured cost: about 21 min
  CPU for all of Part 2.
- **Encoder data.** Oracle frames plus policy and archive rollouts, recorded
  through the existing writer (h5, schema 3 with `location`). Retrain with
  `scripts/train.py` before RL, freeze during RL, and allow one refresh as new
  maps enter the data.
- **Reward** (training only, RAM):
  - +1 per objective completed.
  - +0.01 per new (map, y, x) tile **within the current objective**. The visited
    set resets when an objective completes, which is what makes the backtrack
    pay again.
  - A scaled intrinsic bonus (item 3).
  - No level, heal or battle terms. Prior art shows every one of those got hacked.
- **Resets: a Go-Explore archive.** Training episodes start from save states
  sampled from an archive keyed by (objective, map, coarse tile cell). It is
  seeded with oracle and bootstrap states, and new cells reached by the policy
  are added. Training only.
- **Env.** Vectorised PyBoy workers using `PokeEnv` timing (60 frames per
  action). Handlers run inside the env wrapper: one policy action may expand into
  several handler actions before the next observation. Every action is logged
  with its controller.

## Evaluation

- **Protocol.** One continuous run per seed: no rollback, a fixed total action
  budget that counts handler actions, and 10 policy-sampling seeds plus a greedy run.
  - Stage A starts from the Route 1 state that replaying `play_milestones_s1`
    reaches, (12, 35, 10).
  - Stage B starts from `states/bedroom.state`.
- **Headline.** Per-milestone success over seeds, the **learned share** of
  actions, actions to each milestone, and a GIF of every eval run. A human must
  watch the GIFs, because reward hacks show up there first.
- **Action budget:** 3× the oracle's action count for the same stage (my call).
- **Trained arms** (my call). Every arm gets the same handlers, extrinsic
  reward and budget:
  1. pixels-only PPO (small CNN on raw frames) plus an **RND** bonus, which
     needs no world model. This makes it a strong baseline that is free of the
     world model.
  2. frozen world-model features plus RND
  3. frozen world-model features plus the world-model bonus

  Arm 1 vs 2 isolates the representation. Arm 2 vs 3 isolates the bonus
  source.
- **Cheap baselines, same handlers and budget:**
  - random
  - the 7 constant buttons
  - the BFS oracle as the ceiling, labelled as using rollback
  - for Stage B's Part 1, today's waypoint controller
- **Done** (my call): Stage A reaches map 51 on at least 5 of 10 seeds. Stage B
  does the same from a new game. Report learned share and the ablations either
  way.
- **Kill and report rule.** If pixels-only PPO matches the world-model arms
  within seed noise, that is the result and it gets reported. If no arm
  completes `get_parcel` within the training budget fixed in the plan, stop and
  revisit the reward/archive design before scaling.

## Goals, non-goals, anti-goals

- **Goals:** the play-time contract holds. Part 2 is reached from Route 1 (A),
  then from a new game (B). The world model's contribution is measured against
  a pixels-only ablation.
- **Non-goals:**
  - learning battles
  - Viridian Forest interior, Brock and later parts
  - the Route 22 rival
  - catching Pokémon
  - LLM or VLM components
  - cloud training
  - speed records
  - colour mode (DMG stays)
- **Anti-goals:**
  - finishing with a falling learned share
  - any play-time read of coordinates, map id, waypoints or walls, and any rollback
  - an unwatched reward hack
  - "the world model helps" without the ablation
  - a pass from one seed

## Constraints

- Local Apple M5 (MPS), $0. Measured: encoder inference ~1,758 frames/s;
  world-model training ~403 frames/s; 10-epoch run 64 min. Multi-process
  PyBoy throughput is **unmeasured**, so measure it first.
- Python ≥ 3.12, torch, PyBoy, h5py. CI has no ROM. ROM-touching scripts must
  pass `--help` in CI, and logic gets unit tests with fakes. 210 tests today,
  and they must stay green.
- The ROM stays out of git, and so do `*.ram` saves.

## What is left to the builder

- PPO implementation: hand-rolled in torch, or a dependency. Prefer no heavy new
  dependencies.
- Hyperparameters, worker count, total training budget (fixed in the plan
  after the throughput measurement), and archive cell granularity.
- How play time detects an open text box without rollback: a pixel detector or
  a RAM indicator. The handler is declared harness either way.
- Whether the intrinsic bonus is prediction error or ensemble disagreement.
  Pick after a small comparison.
