---
title: PokeDreamer Yellow
license: mit
task_categories:
- reinforcement-learning
pretty_name: PokeDreamer Yellow — GBC colour trajectories across the whole game
size_categories:
- 100K-1M
---

# PokeDreamer Yellow

500,000 transitions of Pokémon Yellow played in PyBoy. They are written in the
schema of [PokeDreamer v2](https://huggingface.co/datasets/xxxTEMPESTxxx/PokeDreamer)
(Pokémon Red), so a loader written for that dataset reads these chunks unchanged.
Use it to train world models (RSSM / DreamerV3-style) and RL agents.

## Compared with PokeDreamer Red

Per 10,000 steps, PokeDreamer's numbers come from two of its chunks:

| | PokeDreamer Red | **PokeDreamer Yellow** |
|---|---|---|
| Steps | 200,000 | **500,000** |
| Frames | grey (DMG) | **GBC colour** |
| Start states | 5 (Pallet → Pewter) | **235, covering 211 maps** |
| Distinct maps / 10 k | 7.0 | **12.7** |
| Distinct (map, y, x) tiles / 10 k | 514.5 | **763.8** |
| Map changes / 10 k | 28.5 | **36.6** |
| Battle share | 0.117 | 0.208 |
| `dialog_opens` share | 0.958 | 0.265 |
| `rewards` | placeholder 0 | exploration + level + badge + event |

**About PokeDreamer's `dialog_opens`.** It is True on 95.8% of its frames,
including plain overworld frames with no box on screen, so we treat that probe
as broken there. Here it is read from the game's tile buffer: both corner tiles
of the message box must be present. Checked by eye on 3,000 frames.

## Data structure

There are 50 chunks, `transitions_0001.npz` … `transitions_0050.npz`, each
holding 10,000 steps. Every array is indexed by step.

### PokeDreamer's arrays (same names, dtypes and meanings)

| array | dtype, shape | |
|---|---|---|
| `obs` | u1 (N,144,160,3) | RGB, GBC colour as PyBoy draws it |
| `actions` | u1 (N,) | 0 down, 1 left, 2 right, 3 up, 4 a, 5 b, 6 start, 7 pass |
| `rewards` | f4 (N,) | weighted sum of the four components below |
| `episode_starts` | bool (N,) | first step of an episode |
| `map_ids`, `xs`, `ys` | u1 (N,) | wCurMap, wXCoord, wYCoord |
| `facings` | u1 (N,) | 0 down, 4 up, 8 left, 12 right |
| `in_battles` | bool (N,) | wIsInBattle ≠ 0 |
| `dialog_opens` | bool (N,) | message box on screen (always True in battle) |
| `badges` | u1 (N,) | number of badges |
| `party_hps`, `party_max_hps` | i2 (N,6) | 0 past the end of the party |

### Additions

| array | dtype, shape | |
|---|---|---|
| `party_counts` | u1 (N,) | |
| `party_levels` | u1 (N,6) | |
| `event_flag_counts` | u2 (N,) | set bits in wEventFlags |
| `reward_explore` | f4 (N,) | 1 for a (map, y, x) not yet visited this episode |
| `reward_level` | f4 (N,) | increase in summed party level |
| `reward_badge` | f4 (N,) | new badges |
| `reward_event` | f4 (N,) | new event flags |
| `policy_ids` | u1 (N,) | 0 sticky-random, 1 frontier |
| `state_ids` | u2 (N,) | index into `collection.json` → `states` |

`rewards` = 0.01·explore + 1·level + 5·badge + 0.5·event. The components are
stored separately, so you can re-weight them without re-collecting.

### Step semantics

- `obs[t]` and every probe `[t]` are read **before** `actions[t]` is pressed.
- `rewards[t]` is what `actions[t]` earned.
- An action is held for 8 ticks and then released, and each step lasts 24
  ticks in total. Only the last tick is rendered.

## How it was collected

**Story states.** These were reached with no human play. Bootstrap goes from
the bare ROM through the intro, then a BFS oracle walks the story in colour:
Pikachu → rival → Route 1 → Oak's Parcel → Pokédex → Viridian Forest.

**Everywhere else.** Each map's warp table is read out of the ROM through
pret's symbol file. The game's `WarpFound2` routine is hooked so that a warp
the player steps on leads to the chosen map and warp. The game itself then
loads the map, with its NPCs, tileset, palettes and Pikachu following. 228 of
229 attempted teleports arrived with control. Seafoam B4F did not, because its
currents move the player.

Before a state was saved, Pikachu was raised to a level that fits the area
(6–58), and the cast was renamed to the longplay's: RED, BLUE, PIKACHU.

**Policies.** There are 250 episodes of 2,000 steps, dealt round-robin over the
start states. Each episode uses one of two policies, alternating:
- **sticky-random:** a random button, repeated geometrically. Mix: directions
  64%, A 20%, B 10%, START 3%, pass 3%.
- **frontier:** walks towards the least-visited neighbouring tile and learns
  walls from moves that went nowhere. In menus it presses A 60% of the time,
  and leans up/left so it can get back to FIGHT.

Neither policy ever rolls the emulator back.

## Checks (`scripts/check_dreamer.py`)

1. PokeDreamer's README loader runs on all 50 chunks, and every array matches
   the schema.
2. **Replay is exact.** For 12 episodes spread across the set (24,000 steps),
   pressing the recorded actions from the recorded start state reproduced every
   frame and every probe.
3. Every RAM field was checked on the cartridge:
   - Facings turn the sprite.
   - A poked badge byte shows on the save screen.
   - HP, max HP and level match the battle screen.
   - All 14 addresses agree with pret's symbol table.

## Known deviations from real play

- **The teleports are emulator surgery.** Story flags are whatever the Pokédex
  save had, so later towns contain trainers and events out of order.
- **Rock Tunnel is lit.** The game only darkens it when you enter from outdoors.
- **Levels and stats are scaled**, not earned. Pikachu's moves stay early-game.
- **The party is one Pikachu.**

## Reproduce

```
uv run python -m scripts.fetch_symbols
uv run python -m scripts.bootstrap --color --to pikachu --seeds 40 --max-steps 3000
uv run python -m scripts.oracle --color
uv run python -m scripts.make_start_states --workers 8
uv run python -m scripts.collect_dreamer --steps 500000 --workers 8 --pack
uv run python -m scripts.check_dreamer
```

On an M5 this takes about 18 min for the oracle, 70 s for the teleports, 145 s
to collect and 45 s to check. The ROM is not included.
