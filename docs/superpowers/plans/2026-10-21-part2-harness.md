# Part 2 Harness Implementation Plan (Stage 1 of 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the play-time harness the Part 2 design demands — flag-based objectives, declared scripted handlers, a rollback-free control probe, a repo-grade BFS oracle, and a contract-enforcing runner — so that an RL policy can be dropped in as the only missing piece.

**Architecture:** Five new `poke/` modules, each one responsibility, plus three scripts. `poke/flags.py` reads the story out of `wEventFlags` and the bag. `poke/objectives.py` turns those reads into the ordered objective the policy is told it is on, using flags only — never map id or position. `poke/control.py` answers "does the player have control" without a snapshot, because play time bans rollback. `poke/handlers.py` is the declared scripted harness (text, decline, battle, no-control). `poke/contract.py` wraps `PokeEnv` so a play-time read of coordinates, map id or walls raises instead of silently cheating, and tallies the learned share. `scripts/oracle.py` ports the throwaway spike into the repo as the training-time route finder and ceiling.

**Tech Stack:** Python ≥ 3.12, PyBoy, numpy, h5py, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-21-part2-world-model-rl-design.md`

**Why this plan stops short of RL:** the spec fixes the RL training budget "after the throughput measurement" and says multi-process PyBoy throughput "is **unmeasured**, so measure it first". Task 8 produces that number. The RL plan (encoder refresh, PPO, Go-Explore archive, intrinsic bonus, the three arms, the eval protocol) is written against it as `docs/superpowers/plans/<date>-part2-rl.md`. Everything in this plan is working, measurable software on its own: with it the repo can run Part 2 under the contract with the cheap baselines and report the oracle ceiling.

## Global Constraints

- Python ≥ 3.12. Dependencies: torch, PyBoy, h5py, numpy. **Prefer no heavy new dependencies.**
- CI has no ROM. Every ROM-touching script must pass `--help` in CI; all logic gets unit tests with fakes (`tests/conftest.py::FakeEnv`).
- **210 tests pass today and must stay green.** Run `uv run pytest -q` before every commit.
- The ROM stays out of git. So do `*.gb`, `*.gbc`, `*.sav`, `*.state`, `*.ram`, `states/`.
- DMG only. `PokeEnv` forces `cgb=False`; colour mode is a non-goal.
- Yellow WRAM is Red minus one above `$CF1A`. Every new address goes in `poke/games.py::RamMap` with a `confidence` entry, never as a module-level constant.
- **Play time bans emulator save/restore.** Any `snapshot`/`restore` in a play-time path is a bug, not an optimisation.
- **Reward is training-only and never a policy input.**
- Every action is logged with the controller that chose it. Turns where nobody has control and no text box is open are labelled `no_control` and count as neither learned nor handler actions.
- Local Apple M5 (MPS), $0. No cloud training.

## Review Focus

Five things the spec implies that no task's happy path would otherwise exercise. Each has its test pinned to the task that owns the code.

1. **The first input after a warp is swallowed by the fade-in.** The state does not change, so any "no effect → dead end / wall" logic reads every door as a wall. A reasonable person expects walking through a door to work. → Task 4 (`test_handlers.py::test_warp_emits_one_noop_before_anything_else`).
2. **Bit 57 `GOT_OAKS_PARCEL` never clears on delivery.** Treating it as "carrying" makes `deliver_parcel` permanently active and `enter_forest` unreachable. Carrying is a bag read. → Task 1 (`test_flags.py::test_carrying_is_the_bag_not_the_flag`) and Task 2 (`test_objectives.py::test_delivery_advances_even_though_bit_57_stays_set`).
3. **Blacking out warps the player home or to a Pokémon Center mid-run.** The spec says blackout is not special-cased and the run continues. A per-objective visited set keyed on position must survive a discontinuous jump rather than throw or wedge. → Task 6 (`test_contract.py::test_blackout_warp_is_logged_and_does_not_reset_the_objective`).
4. **A Red address read against a Yellow ROM returns a plausible byte.** Every address added here is above `$CF1A` except `facing`, so the shift applies to all but one. → Task 1 (`test_flags.py::test_yellow_event_flags_are_red_minus_one_and_facing_is_not`).
5. **One 60-frame action sometimes walks two tiles.** Anything that assumes consecutive observations are 4-adjacent — tile counting, dedup, "did it move" — is wrong on those steps. → Task 6 (`test_contract.py::test_two_tile_step_counts_as_one_action_and_two_new_tiles`).

---

## File Structure

| File | Responsibility |
|---|---|
| `poke/games.py` (modify) | add `event_flags`, `num_bag_items`, `facing`, `text_box` to `RamMap` for both versions, with confidence |
| `poke/flags.py` (create) | event-flag bits, bag reads, named story constants |
| `poke/objectives.py` (create) | the Part 2 objective chain; flags-only selector, map-allowed scorer |
| `poke/control.py` (create) | rollback-free "does the player have control" |
| `poke/handlers.py` (create) | the declared scripted harness: text, decline, battle, warp no-op |
| `poke/contract.py` (create) | play-time guard around `PokeEnv` + controller accounting |
| `poke/oracle.py` (create) | BFS over (map, y, x, facing, story signature) with snapshots — training only |
| `scripts/calibrate_control.py` (create) | measure which WRAM byte tracks control; pin the winner |
| `scripts/oracle.py` (create) | CLI: run the oracle, write milestone states and demo episodes |
| `scripts/play2.py` (create) | contract-compliant Part 2 runner: random, constant, oracle-replay policies |
| `scripts/bench_workers.py` (create) | multi-process PyBoy throughput |
| `tests/test_flags.py`, `test_objectives.py`, `test_control.py`, `test_handlers.py`, `test_contract.py`, `test_oracle.py`, `test_play2.py`, `test_bench_workers.py` (create) | unit tests, all with fakes |

---

### Task 1: Story RAM — event flags and the bag

**Files:**
- Modify: `poke/games.py:28-38` (`RamMap`), `poke/games.py:52-105` (`RED`, `YELLOW`)
- Create: `poke/flags.py`
- Test: `tests/test_flags.py`

**Interfaces:**
- Consumes: `poke.games.RamMap`, `tests.conftest.FakeEnv`.
- Produces: `poke.flags.event_bit(env, index) -> int`, `bag_items(env) -> list[tuple[int, int]]`, `carrying_parcel(env) -> bool`, and the module constants `GOT_STARTER=34`, `BATTLED_RIVAL_IN_OAKS_LAB=35`, `GOT_POKEDEX=37`, `OAK_APPEARED_IN_PALLET=39`, `COMPLETED_CATCH_TRAINING=45`, `INITIAL_CATCH_TRAINING=47`, `OAK_GOT_PARCEL=56`, `GOT_OAKS_PARCEL=57`, `OAKS_PARCEL=0x46`, `NUM_EVENT_BYTES=320`, `BAG_CAPACITY=20`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_flags.py
import pytest

from poke.flags import (
    GOT_OAKS_PARCEL, GOT_POKEDEX, OAKS_PARCEL, OAK_GOT_PARCEL,
    bag_items, carrying_parcel, event_bit,
)
from poke.games import RED, YELLOW
from tests.conftest import FakeEnv


def env_with(flags=(), bag=(), game=YELLOW):
    """FakeEnv whose event bytes carry `flags` and whose bag carries `bag`."""
    ram = {}
    for index in flags:
        addr = game.ram.event_flags + index // 8
        ram[addr] = ram.get(addr, 0) | (1 << (index % 8))
    ram[game.ram.num_bag_items] = len(bag)
    for slot, (item, qty) in enumerate(bag):
        ram[game.ram.num_bag_items + 1 + 2 * slot] = item
        ram[game.ram.num_bag_items + 2 + 2 * slot] = qty
    return FakeEnv(game, ram)


def test_event_bit_reads_the_right_byte_and_bit():
    env = env_with(flags=[GOT_POKEDEX])
    assert event_bit(env, GOT_POKEDEX) == 1
    assert event_bit(env, GOT_OAKS_PARCEL) == 0
    # Bit 37 is byte 4, bit 5 -- 0xD746 + 4 == 0xD74A, as measured.
    assert YELLOW.ram.event_flags + GOT_POKEDEX // 8 == 0xD74A
    assert YELLOW.ram.event_flags + OAK_GOT_PARCEL // 8 == 0xD74D


def test_unreadable_ram_is_zero_not_a_crash():
    assert event_bit(FakeEnv(YELLOW, {}), GOT_POKEDEX) == 0


def test_bag_items_stop_at_the_count():
    env = env_with(bag=[(OAKS_PARCEL, 1), (0x14, 3)])
    assert bag_items(env) == [(OAKS_PARCEL, 1), (0x14, 3)]


def test_absurd_bag_count_reads_as_empty_rather_than_walking_wram():
    env = FakeEnv(YELLOW, {YELLOW.ram.num_bag_items: 200})
    assert bag_items(env) == []


def test_carrying_is_the_bag_not_the_flag():
    """Bit 57 never clears on delivery, so a flag-based 'carrying' would be
    true forever and the delivery objective would never end. Measured on the
    cartridge: after Oak takes it, bit 57 stays 1 and the bag is empty."""
    delivered = env_with(flags=[GOT_OAKS_PARCEL, OAK_GOT_PARCEL], bag=[])
    assert event_bit(delivered, GOT_OAKS_PARCEL) == 1
    assert not carrying_parcel(delivered)
    assert carrying_parcel(env_with(flags=[GOT_OAKS_PARCEL], bag=[(OAKS_PARCEL, 1)]))


def test_yellow_event_flags_are_red_minus_one_and_facing_is_not():
    """Everything above $CF1A shifts by one in Yellow; facing is below it."""
    assert YELLOW.ram.event_flags == RED.ram.event_flags - 1
    assert YELLOW.ram.num_bag_items == RED.ram.num_bag_items - 1
    assert YELLOW.ram.facing == RED.ram.facing == 0xC109
    assert YELLOW.ram.facing < 0xCF1A < YELLOW.ram.event_flags


def test_every_new_yellow_address_declares_its_confidence():
    for name in ("event_flags", "num_bag_items", "facing"):
        assert name in YELLOW.ram.confidence
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_flags.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'poke.flags'`

- [ ] **Step 3: Extend `RamMap` and both registries**

In `poke/games.py`, add four fields to `RamMap` after `in_battle`:

```python
    event_flags: int
    num_bag_items: int
    facing: int
    text_box: int | None = None
```

In `RED.ram`, add:

```python
        event_flags=0xD747,
        num_bag_items=0xD31D,
        facing=0xC109,
```

and in its `confidence` dict:

```python
            "event_flags": "documented",
            "num_bag_items": "documented",
            "facing": "documented",
```

In `YELLOW.ram`, add:

```python
        event_flags=0xD746,
        num_bag_items=0xD31C,
        facing=0xC109,
```

and in its `confidence` dict:

```python
            # Watched bits 37/56/57 flip in the right order on this cartridge
            # while the parcel moved through the bag (flag spike, 2026-10-21).
            "event_flags": "confirmed-on-rom",
            "num_bag_items": "confirmed-on-rom",
            # Below $CF1A, so the -1 shift does not apply.
            "facing": "documented",
```

Leave `text_box` unset for both; Task 3 fills it from a measurement.

- [ ] **Step 4: Write `poke/flags.py`**

```python
"""The story, read out of RAM: event flags and the bag.

`poke.ram` covers the things a frame-by-frame controller needs (position,
party, battle). This module covers the things the *plot* is recorded in, which
is what decides which objective the player is on.

Bit indices follow pret/pokeyellow's event_constants. The ones below were
watched flipping on this cartridge, in order, during the parcel errand.
"""

from __future__ import annotations

GOT_STARTER = 34
BATTLED_RIVAL_IN_OAKS_LAB = 35
GOT_POKEBALLS_FROM_OAK = 36
GOT_POKEDEX = 37
OAK_APPEARED_IN_PALLET = 39
SPAWNED_OLD_MAN_1 = 44
COMPLETED_CATCH_TRAINING = 45
INITIAL_CATCH_TRAINING = 47
OAK_GOT_PARCEL = 56
GOT_OAKS_PARCEL = 57

NUM_EVENT_BYTES = 320
OAKS_PARCEL = 0x46
BAG_CAPACITY = 20


def event_bit(env, index: int) -> int:
    """One bit of wEventFlags. An unreadable byte reads as 0, not as a crash:
    a missing read means "this has not happened", which is the safe answer."""
    byte = env.read_ram(env.game.ram.event_flags + index // 8)
    return 0 if byte is None else (byte >> (index % 8)) & 1


def bag_items(env) -> list[tuple[int, int]]:
    """(item id, quantity) pairs. The count byte is trusted only up to the
    bag's real capacity -- a wrong address reads a huge count and would
    otherwise march off through WRAM inventing items."""
    base = env.game.ram.num_bag_items
    count = env.read_ram(base)
    if count is None or count > BAG_CAPACITY:
        return []
    pairs = []
    for slot in range(count):
        item = env.read_ram(base + 1 + 2 * slot)
        qty = env.read_ram(base + 2 + 2 * slot)
        if item is None:
            break
        pairs.append((int(item), 0 if qty is None else int(qty)))
    return pairs


def carrying_parcel(env) -> bool:
    """Whether Oak's Parcel is in the bag right now.

    Not `event_bit(env, GOT_OAKS_PARCEL)`: measured on the cartridge, bit 57
    never clears when Oak takes the parcel. The bit says "was collected"; the
    bag says "is held".
    """
    return any(item == OAKS_PARCEL for item, _ in bag_items(env))
```

- [ ] **Step 5: Run the new test and the whole suite**

Run: `uv run pytest tests/test_flags.py -q && uv run pytest -q`
Expected: new tests PASS; total count rises from 210 with no failures.

- [ ] **Step 6: Commit**

```bash
git add poke/flags.py poke/games.py tests/test_flags.py
git commit -m "Read the story out of RAM: event flags and the bag

Yellow's wEventFlags (0xD746) and wNumBagItems (0xD31C) are Red minus one;
facing (0xC109) is below \$CF1A and is not shifted. Bits 37/56/57 were watched
flipping in order on this cartridge.

Carrying the parcel is a bag read, not bit 57: the bit never clears when Oak
takes it, so a flag-based check would strand the delivery objective.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: The Part 2 objective chain

**Files:**
- Create: `poke/objectives.py`
- Test: `tests/test_objectives.py`

**Interfaces:**
- Consumes: `poke.flags` (Task 1), `poke.ram.cur_map`, `poke.maps`.
- Produces: `poke.objectives.Objective` (frozen dataclass with `key: str`, `index: int`, `description: str`, `active: Callable[[object], bool]`, `reached: Callable[[object], bool]`), `CHAIN: tuple[Objective, ...]`, `current(env) -> Objective`, `one_hot(env) -> list[float]`, `N_OBJECTIVES: int`, `get(key) -> Objective`.

**Contract note for the implementer:** `active` is the policy's input, so it may read **only** event flags and the bag. `reached` judges the run, so it may read the map. Nothing in `active` may call `cur_map`, `coords`, or touch a walkmap. The test below enforces this with a RAM-access recorder; do not weaken it.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_objectives.py
import pytest

from poke.flags import (
    BATTLED_RIVAL_IN_OAKS_LAB, GOT_OAKS_PARCEL, GOT_POKEDEX,
    OAKS_PARCEL, OAK_APPEARED_IN_PALLET, OAK_GOT_PARCEL,
)
from poke.games import YELLOW
from poke.objectives import CHAIN, N_OBJECTIVES, current, get, one_hot
from tests.conftest import FakeEnv

VIRIDIAN_FOREST = 51


class RecordingEnv(FakeEnv):
    """FakeEnv that remembers every address read, so a test can prove that a
    predicate never looked at the player's position."""

    def __init__(self, game, ram=None):
        super().__init__(game, ram)
        self.reads = []

    def read_ram(self, addr):
        self.reads.append(addr)
        return super().read_ram(addr)


def env_with(flags=(), bag=(), map_id=None):
    ram = {}
    for index in flags:
        addr = YELLOW.ram.event_flags + index // 8
        ram[addr] = ram.get(addr, 0) | (1 << (index % 8))
    ram[YELLOW.ram.num_bag_items] = len(bag)
    for slot, (item, qty) in enumerate(bag):
        ram[YELLOW.ram.num_bag_items + 1 + 2 * slot] = item
        ram[YELLOW.ram.num_bag_items + 2 + 2 * slot] = qty
    if map_id is not None:
        ram[YELLOW.ram.cur_map] = map_id
    return RecordingEnv(YELLOW, ram)


def test_chain_is_the_specs_order():
    assert [o.key for o in CHAIN] == [
        "trigger_oak", "get_starter", "get_parcel",
        "deliver_parcel", "enter_forest",
    ]
    assert [o.index for o in CHAIN] == [0, 1, 2, 3, 4]
    assert N_OBJECTIVES == 5


def test_new_game_is_on_trigger_oak():
    assert current(env_with()).key == "trigger_oak"


def test_oak_appeared_moves_to_the_starter():
    assert current(env_with(flags=[OAK_APPEARED_IN_PALLET])).key == "get_starter"


def test_rival_beaten_moves_to_the_parcel_errand():
    env = env_with(flags=[OAK_APPEARED_IN_PALLET, BATTLED_RIVAL_IN_OAKS_LAB])
    assert current(env).key == "get_parcel"


def test_parcel_in_hand_moves_to_delivery():
    env = env_with(
        flags=[OAK_APPEARED_IN_PALLET, BATTLED_RIVAL_IN_OAKS_LAB, GOT_OAKS_PARCEL],
        bag=[(OAKS_PARCEL, 1)],
    )
    assert current(env).key == "deliver_parcel"


def test_delivery_advances_even_though_bit_57_stays_set():
    """Bit 57 is still 1 after Oak takes the parcel. If `deliver_parcel` keyed
    off 57 the run would sit on it forever and never aim at the forest."""
    env = env_with(
        flags=[OAK_APPEARED_IN_PALLET, BATTLED_RIVAL_IN_OAKS_LAB,
               GOT_OAKS_PARCEL, OAK_GOT_PARCEL, GOT_POKEDEX],
        bag=[],
    )
    assert current(env).key == "enter_forest"


def test_the_policys_objective_never_reads_position():
    """`active` is a policy input, so it may read flags and the bag only."""
    forbidden = {YELLOW.ram.cur_map, YELLOW.ram.y_coord, YELLOW.ram.x_coord}
    for objective in CHAIN:
        env = env_with(map_id=VIRIDIAN_FOREST)
        objective.active(env)
        assert not (set(env.reads) & forbidden), objective.key


def test_reached_may_read_the_map_because_it_only_scores():
    env = env_with(map_id=VIRIDIAN_FOREST)
    assert get("enter_forest").reached(env)
    assert not get("enter_forest").reached(env_with(map_id=50))


def test_one_hot_is_the_policy_input():
    vector = one_hot(env_with(flags=[OAK_APPEARED_IN_PALLET]))
    assert vector == [0.0, 1.0, 0.0, 0.0, 0.0]
    assert len(vector) == N_OBJECTIVES


def test_unknown_objective_raises():
    with pytest.raises(ValueError, match="unknown objective"):
        get("catch_mewtwo")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_objectives.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'poke.objectives'`

- [ ] **Step 3: Write `poke/objectives.py`**

```python
"""What the player is trying to do right now, read from the story flags.

The play-time contract lets the policy be told its current objective, and
nothing else about where it is. So the selector below reads event flags and
the bag and never the map or the coordinates: `active` is a policy input.
`reached` is allowed the map, because it scores a finished run rather than
steering a live one.

The chain runs from a new game to Viridian Forest. Leaving Pallet is not an
objective of its own -- it is folded into `get_parcel`, which is what makes the
Route 1 walk pay.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from poke.flags import (
    BATTLED_RIVAL_IN_OAKS_LAB, GOT_OAKS_PARCEL, GOT_POKEDEX,
    OAK_APPEARED_IN_PALLET, OAK_GOT_PARCEL, carrying_parcel, event_bit,
)
from poke.ram import cur_map

VIRIDIAN_FOREST = 51


@dataclass(frozen=True)
class Objective:
    key: str
    index: int
    description: str
    active: Callable[[object], bool]
    reached: Callable[[object], bool]


def _oak_not_yet(env):
    return event_bit(env, OAK_APPEARED_IN_PALLET) == 0


def _oak_appeared(env):
    return event_bit(env, OAK_APPEARED_IN_PALLET) == 1


def _rival_done(env):
    return event_bit(env, BATTLED_RIVAL_IN_OAKS_LAB) == 1


def _parcel_collected(env):
    return event_bit(env, GOT_OAKS_PARCEL) == 1


def _parcel_delivered(env):
    return event_bit(env, OAK_GOT_PARCEL) == 1


CHAIN: tuple[Objective, ...] = (
    Objective(
        key="trigger_oak",
        index=0,
        description="get Oak to appear and follow him into the lab",
        active=_oak_not_yet,
        reached=_oak_appeared,
    ),
    Objective(
        key="get_starter",
        index=1,
        description="take Pikachu and survive the rival fight",
        active=lambda env: _oak_appeared(env) and not _rival_done(env),
        reached=_rival_done,
    ),
    Objective(
        key="get_parcel",
        index=2,
        # Leaving Pallet and the whole of Route 1 live inside this one.
        description="cross Route 1 to Viridian and pick up Oak's Parcel",
        active=lambda env: _rival_done(env) and not _parcel_collected(env),
        reached=_parcel_collected,
    ),
    Objective(
        key="deliver_parcel",
        index=3,
        description="carry the parcel back to Oak and take the Pokedex",
        active=lambda env: _parcel_collected(env) and not _parcel_delivered(env),
        reached=lambda env: _parcel_delivered(env) and event_bit(env, GOT_POKEDEX) == 1,
    ),
    Objective(
        key="enter_forest",
        index=4,
        description="north past the old man into Viridian Forest",
        active=_parcel_delivered,
        reached=lambda env: cur_map(env) == VIRIDIAN_FOREST,
    ),
)

N_OBJECTIVES = len(CHAIN)


def current(env) -> Objective:
    """The first objective whose preconditions hold. Flags and bag only.

    The predicates are written to be mutually exclusive, but "first match"
    rather than "the one match" is deliberate: a save state captured mid-scene
    should pick the earliest plausible objective rather than raise.
    """
    for objective in CHAIN:
        if objective.active(env):
            return objective
    return CHAIN[-1]


def one_hot(env) -> list[float]:
    vector = [0.0] * N_OBJECTIVES
    vector[current(env).index] = 1.0
    return vector


def get(key: str) -> Objective:
    for objective in CHAIN:
        if objective.key == key:
            return objective
    known = ", ".join(o.key for o in CHAIN)
    raise ValueError(f"unknown objective {key!r}; known: {known}")
```

Note: `carrying_parcel` is imported for the implementer's use in a later task (the reward's "is the errand live" check) and is not used by the chain; delete the import if your linter objects.

- [ ] **Step 4: Run the test**

Run: `uv run pytest tests/test_objectives.py -q && uv run pytest -q`
Expected: PASS, suite still green.

- [ ] **Step 5: Commit**

```bash
git add poke/objectives.py tests/test_objectives.py
git commit -m "The objective the policy is told it is on, from flags alone

Five objectives from a new game to Viridian Forest. \`active\` is a policy
input and reads event flags and the bag only -- a test records every RAM
access and fails if a predicate touches cur_map, y or x. \`reached\` scores a
run and may read the map.

Leaving Pallet is folded into get_parcel, so the Route 1 walk pays once.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: A control probe that does not roll back

**Files:**
- Create: `scripts/calibrate_control.py`, `poke/control.py`
- Modify: `poke/games.py` (fill `text_box` for YELLOW from the measurement)
- Test: `tests/test_control.py`

**Why:** today's `has_control` in `scripts/play.py:373` snapshots and restores. Play time bans that. The replacement has to be a RAM read, and which byte it is must be **measured on this cartridge**, not guessed — that is how every other address in this repo was settled.

**Interfaces:**
- Consumes: `poke.games.RamMap.text_box`, `poke.ram`.
- Produces: `poke.control.has_control(env) -> bool`, `poke.control.text_box_open(env) -> bool`, `poke.control.CANDIDATES: tuple[int, ...]`, and `scripts/calibrate_control.py` writing `results/control_calibration.json` with `{"address": int, "agreement": float, "samples": int, "rejected": [...]}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_control.py
import pytest

from poke.control import has_control, text_box_open
from poke.games import YELLOW
from poke.ram import NO_BATTLE, WILD_BATTLE
from tests.conftest import FakeEnv


def env_at(text_box=0, battle=NO_BATTLE):
    return FakeEnv(YELLOW, {
        YELLOW.ram.text_box: text_box,
        YELLOW.ram.in_battle: battle,
    })


def test_control_is_free_when_no_box_is_open():
    assert has_control(env_at(text_box=0))
    assert not text_box_open(env_at(text_box=0))


def test_an_open_box_takes_control_away():
    assert not has_control(env_at(text_box=1))
    assert text_box_open(env_at(text_box=1))


def test_a_battle_is_not_control_even_with_no_box():
    """The battle handler owns these turns; the policy must not be asked for
    an overworld button while a fight is running."""
    assert not has_control(env_at(text_box=0, battle=WILD_BATTLE))


def test_an_unreadable_byte_is_not_control():
    """Guessing 'yes, you can walk' on a failed read hands the policy turns it
    cannot use. The safe default is to let a handler take the turn."""
    assert not has_control(FakeEnv(YELLOW, {}))


def test_the_probe_never_snapshots():
    """Play time bans rollback, so the probe must not be able to use it."""
    class NoRollbackEnv(FakeEnv):
        def snapshot(self):
            raise AssertionError("has_control must not snapshot at play time")

        def restore(self, blob):
            raise AssertionError("has_control must not restore at play time")

    assert has_control(NoRollbackEnv(YELLOW, {YELLOW.ram.text_box: 0}))


def test_yellow_declares_a_measured_text_box_address():
    assert YELLOW.ram.text_box is not None
    assert YELLOW.ram.confidence["text_box"] == "confirmed-on-rom"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_control.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'poke.control'`

- [ ] **Step 3: Write the calibration script**

```python
# scripts/calibrate_control.py
"""Find the WRAM byte that tracks "the player has control".

`scripts/play.py` answers that question today by snapshotting, pressing two
different directions and comparing screens. It is correct and it is banned at
play time, because the contract forbids rollback. So: run the game, take the
expensive answer as ground truth, and look for a byte that agrees with it.

Same shape as scripts/calibrate_ram.py -- scan, score, print, pin by hand.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from poke.actions import ACTIONS
from poke.env import PokeEnv

WRAM = range(0xC000, 0xE000)


def ground_truth(env) -> bool:
    """The rollback answer: two buttons from one state, two different screens."""
    blob = env.snapshot()
    screens = []
    try:
        for button in ("down", "right"):
            env.restore(blob)
            env.step(ACTIONS.index(button))
            screens.append(env.screen()[:, :, 0].tobytes())
    finally:
        env.restore(blob)
    return screens[0] != screens[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rom", default="pokeyellow.gbc")
    parser.add_argument("--state", default="states/bedroom.state")
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/control_calibration.json")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    env = PokeEnv(args.rom, args.state)
    env.reset()

    agree = defaultdict(int)
    without_control_values = defaultdict(lambda: defaultdict(int))
    seen_true = seen_false = 0
    for _ in range(args.samples):
        truth = ground_truth(env)
        seen_true += truth
        seen_false += not truth
        for addr in WRAM:
            value = env.read_ram(addr)
            if (value == 0) == truth:
                agree[addr] += 1
            if not truth:
                without_control_values[addr][value] += 1
        env.step(rng.randrange(len(ACTIONS)))

    ranked = sorted(agree.items(), key=lambda kv: -kv[1])
    best, hits = ranked[0]
    report = {
        "address": best,
        "address_hex": f"{best:#06x}",
        "agreement": hits / args.samples,
        "samples": args.samples,
        "with_control": seen_true,
        "without_control": seen_false,
        # What the winner reads when the player has no control. More than one
        # non-zero value means the byte can tell an open text box from a
        # walking cutscene, which is what the no_control label needs.
        "values_without_control": {
            str(value): count
            for value, count in sorted(without_control_values[best].items(),
                                       key=lambda kv: -kv[1])
        },
        "runners_up": [{"address": f"{a:#06x}", "agreement": h / args.samples}
                       for a, h in ranked[1:11]],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "runners_up"}, indent=1))
    if seen_true < 20 or seen_false < 20:
        print("WARNING: one class is nearly empty; the winner is not evidence. "
              "Re-run from a state where text boxes happen.")
    if report["agreement"] < 0.99:
        print("WARNING: best agreement is under 99%. Do not pin this address.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the calibration on the cartridge**

Run: `uv run python -m scripts.calibrate_control --samples 300`
Expected: a winning address with ≥ 0.99 agreement and at least 20 samples in each class. If no address clears 0.99, widen `--samples`, and start from `states/pikachu.state` as well so that text boxes and battles are represented. **Do not proceed on a warning.** Record the number; it goes in the commit message.

**Then read `values_without_control` and settle the `no_control` label.** The spec wants walking cutscenes and the swallowed post-warp input counted as neither learned nor handler work. A single byte can only separate them from an open text box if it takes more than one non-zero value when control is gone:

- **More than one value:** the dominant value is "a box is open". Record the other value(s) in the report, and in Task 4 return `("no_control", NOOP)` instead of `("text", A)` when the byte reads one of them.
- **One value only:** the byte cannot tell them apart. Write that finding into the report and leave Task 4 as written — cutscene turns are then labelled `text`, which counts them as handler work. That is the conservative direction: it can only *lower* the reported learned share, never inflate it. Say so in the README when the share is reported.

Only the post-warp no-op is labelled `no_control` in the one-value case, and Task 4 already handles that one from `HandlerState.warped` rather than from RAM.

- [ ] **Step 5: Pin the measured address**

In `poke/games.py`, set `text_box=<the measured address>` on `YELLOW.ram` and add to its confidence dict:

```python
            # scripts/calibrate_control.py, 300 samples: agrees with the
            # snapshot-and-compare probe on <N>% of turns.
            "text_box": "confirmed-on-rom",
```

Leave `RED.ram.text_box` as `None`: it has not been measured, and `None` is what makes `has_control` refuse rather than read a wrong byte.

- [ ] **Step 6: Write `poke/control.py`**

```python
"""Does the player have control, without rolling the emulator back.

`scripts/play.py` answers this by snapshotting, pressing two directions and
comparing screens. That is exact, and at play time it is banned: the contract
is one continuous run with no save states. This module is the replacement, and
it is a plain RAM read whose address was measured against the expensive answer
by scripts/calibrate_control.py.
"""

from __future__ import annotations

from poke.ram import in_battle


def text_box_open(env) -> bool:
    """True while a text box or menu owns the input.

    A version with no measured address, or a failed read, answers True: the
    conservative direction is to hand the turn to the declared handler rather
    than to ask the policy for a button the game will swallow.
    """
    addr = env.game.ram.text_box
    if addr is None:
        return True
    value = env.read_ram(addr)
    return True if value is None else value != 0


def has_control(env) -> bool:
    """True when an overworld button from the policy would actually do
    something: no box open, no battle running."""
    return not text_box_open(env) and not in_battle(env)
```

- [ ] **Step 7: Run the tests**

Run: `uv run pytest tests/test_control.py -q && uv run pytest -q`
Expected: PASS, suite green.

- [ ] **Step 8: Commit**

```bash
git add poke/control.py poke/games.py scripts/calibrate_control.py tests/test_control.py results/control_calibration.json
git commit -m "Measure the control probe instead of rolling back for it

Play time bans save states, and has_control() in play.py snapshots twice per
call. scripts/calibrate_control.py scans WRAM against that expensive answer and
reports the byte that agrees with it; the winner is pinned in games.py with its
agreement rate, like every other address in this repo.

Red has no measured address, so text_box stays None there and the probe refuses
rather than reading the wrong byte.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: The declared scripted handlers

**Files:**
- Create: `poke/handlers.py`
- Test: `tests/test_handlers.py`

**Interfaces:**
- Consumes: `poke.actions.ACTIONS`, `poke.control.has_control`/`text_box_open`, `poke.ram.in_battle`, `poke.flags`.
- Produces: `poke.handlers.Handler` protocol with `name: str` and `act(env, state) -> int | None`; concrete `WarpHandler`, `BattleHandler`, `TextHandler`, `DeclineHandler`; `HANDLERS: tuple[Handler, ...]` in priority order; `poke.handlers.HandlerState` (mutable: `warped: bool`, `a_run: int`, `decline_left: int`); `choose(env, state) -> tuple[int, str] | None` returning `(action_index, controller_name)` or `None` when the policy should act.

**Behaviour the handlers own, and nothing else:** advancing an open box, declining (no nickname, no purchases), mashing A through a battle, and spending one no-op after a warp. They never make a story choice.

**Depends on Task 3's measurement.** If `values_without_control` showed the control byte takes a distinct value during walking cutscenes, add that branch to `choose` before the text branch, returning `(NOOP, "no_control")`, and add a test for it alongside the ones below. If it showed one value only, the code below is final and cutscene turns are labelled `text`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_handlers.py
from poke.actions import ACTIONS
from poke.games import YELLOW
from poke.handlers import HandlerState, choose
from poke.ram import NO_BATTLE, TRAINER_BATTLE, WILD_BATTLE
from tests.conftest import FakeEnv

A = ACTIONS.index("a")
DOWN = ACTIONS.index("down")
NOOP = ACTIONS.index("noop")


def env_at(text_box=0, battle=NO_BATTLE, party=1):
    return FakeEnv(YELLOW, {
        YELLOW.ram.text_box: text_box,
        YELLOW.ram.in_battle: battle,
        YELLOW.ram.party_count: party,
    })


def test_free_overworld_turn_belongs_to_the_policy():
    assert choose(env_at(), HandlerState()) is None


def test_an_open_box_is_advanced_with_a():
    action, who = choose(env_at(text_box=1), HandlerState())
    assert (action, who) == (A, "text")


def test_a_battle_is_mashed_with_a():
    action, who = choose(env_at(battle=WILD_BATTLE), HandlerState())
    assert (action, who) == (A, "battle")
    action, who = choose(env_at(battle=TRAINER_BATTLE), HandlerState())
    assert (action, who) == (A, "battle")


def test_battle_outranks_text_so_a_fight_is_never_mislabelled():
    action, who = choose(env_at(text_box=1, battle=WILD_BATTLE), HandlerState())
    assert who == "battle"


def test_warp_emits_one_noop_before_anything_else():
    """The first input after a warp is swallowed by the fade-in: the state does
    not change, and every 'no effect means wall' rule then reads the door it
    just walked through as a dead end. One no-op gives control back."""
    state = HandlerState()
    state.warped = True
    action, who = choose(env_at(), state)
    assert (action, who) == (NOOP, "no_control")
    assert not state.warped
    assert choose(env_at(), state) is None


def test_a_long_run_of_a_switches_to_down_a_to_decline():
    """A answers YES to 'give a nickname?', and inside the naming screen A types
    a letter and B deletes one. The way past is down then A, alternating."""
    state = HandlerState(a_run=8, decline_left=4)
    first, who = choose(env_at(text_box=1), state)
    assert (first, who) == (DOWN, "decline")
    second, _ = choose(env_at(text_box=1), state)
    assert second == A


def test_decline_runs_out_and_hands_back_to_plain_text():
    state = HandlerState(a_run=8, decline_left=1)
    choose(env_at(text_box=1), state)
    action, who = choose(env_at(text_box=1), state)
    assert (action, who) == (A, "text")


def test_handlers_never_choose_a_direction_in_a_battle():
    state = HandlerState(a_run=99, decline_left=9)
    action, who = choose(env_at(battle=WILD_BATTLE), state)
    assert (action, who) == (A, "battle")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_handlers.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'poke.handlers'`

- [ ] **Step 3: Write `poke/handlers.py`**

```python
"""The declared scripted harness.

The play-time contract says navigation and the decision to talk are learned,
and that four things are not: advancing a box that is already open, declining
(no nickname, no purchases), fighting, and the swallowed input after a warp.
Those live here, in one file, so that "what is scripted" is a thing you can
read rather than a thing you have to trace.

Each handler returns an action index or None. `choose` runs them in priority
order and returns None when the turn belongs to the policy. Every action it
returns carries the handler's name, which is what lands in the log and what
the learned share is computed against.
"""

from __future__ import annotations

from dataclasses import dataclass

from poke.actions import ACTIONS
from poke.control import text_box_open
from poke.ram import in_battle

A = ACTIONS.index("a")
DOWN = ACTIONS.index("down")
NOOP = ACTIONS.index("noop")

# How many A presses in a row mean A is not getting anywhere. Measured in
# Part 1: the nickname screen swallowed 207 of them.
TALK_RUN = 8
# How many down/A pairs to spend getting out of a menu once that happens.
DECLINE_BUDGET = 24


@dataclass
class HandlerState:
    """What the handlers remember between turns. Reset per episode."""

    warped: bool = False
    a_run: int = 0
    decline_left: int = 0

    def note(self, action_index: int, moved_map: bool) -> None:
        self.a_run = self.a_run + 1 if action_index == A else 0
        if moved_map:
            self.warped = True


def choose(env, state: HandlerState) -> tuple[int, str] | None:
    """The action a handler claims this turn, or None to let the policy act."""
    if in_battle(env):
        # Mashing A selects the first move, which is Thundershock. A blackout
        # is not special-cased: the run continues from wherever the game puts
        # the player and pays for it in the action count.
        return A, "battle"
    if state.warped:
        # The fade-in eats exactly one input and leaves the state unchanged.
        state.warped = False
        return NOOP, "no_control"
    if not text_box_open(env):
        return None
    if state.decline_left <= 0 and state.a_run >= TALK_RUN:
        state.decline_left = DECLINE_BUDGET
    if state.decline_left > 0:
        state.decline_left -= 1
        # down then A answers NO to a yes/no box; inside a naming screen the
        # same pair walks the cursor off the letter grid to END.
        return (DOWN if state.decline_left % 2 else A), "decline"
    return A, "text"
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_handlers.py -q && uv run pytest -q`
Expected: PASS, suite green.

- [ ] **Step 5: Commit**

```bash
git add poke/handlers.py tests/test_handlers.py
git commit -m "One file that says exactly what is scripted

Battle, warp no-op, text advance and decline, in priority order, each returning
its own controller name. Nothing here makes a story choice.

The warp no-op is the one that matters: the first input after a door is eaten
by the fade-in, and without a turn spent on it every 'no effect means wall'
rule reads the door as a dead end.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: The oracle, in the repo

**Files:**
- Create: `poke/oracle.py`, `scripts/oracle.py`
- Test: `tests/test_oracle.py`

**Source:** the throwaway spike at `/private/tmp/claude-502/-Users-aviad-Documents/a2ddb5c3-1465-49d7-bbb9-ddc9c2787a40/scratchpad/flag_spike/spike.py`. It is **training-time only** — it snapshots and restores freely, which play time bans. Measured: phase A + gate + B + C ≈ 21 min CPU, ending at `(50, 7, 4)`.

**Interfaces:**
- Consumes: `poke.actions.ACTIONS`, `poke.env.PokeEnv`, `poke.flags`, `poke.objectives`.
- Produces: `poke.oracle.search(env, goal, allowed, buttons=..., with_facing=True, max_nodes=12000) -> SearchResult`; `SearchResult` (frozen dataclass: `found: bool`, `path: list[str]`, `expanded: int`, `nodes: int`, `emulated_actions: int`, `maps_seen: dict[int, int]`, `seconds: float`, `end: tuple[int, int, int] | None`); `poke.oracle.story_key(env, with_facing) -> tuple`; `poke.oracle.transition(env, button, cap=400) -> tuple[int, bool] | None`.

- [ ] **Step 1: Write the failing test**

The search has to be testable without a ROM, so `search` takes any object with `snapshot`, `restore`, `step`, `read_ram` and `game`. The test drives it over a hand-built 1-D corridor.

```python
# tests/test_oracle.py
import pytest

from poke.games import YELLOW
from poke.oracle import search, story_key
from tests.conftest import FakeEnv

CORRIDOR = 6


class CorridorEnv(FakeEnv):
    """A one-room, one-axis world: x moves on left/right, walls at both ends.

    Enough to exercise the search's bookkeeping -- snapshots, dedup, the path
    it reconstructs -- without a cartridge.
    """

    def __init__(self):
        super().__init__(YELLOW, {
            YELLOW.ram.cur_map: 1,
            YELLOW.ram.y_coord: 0,
            YELLOW.ram.x_coord: 0,
            YELLOW.ram.in_battle: 0,
            YELLOW.ram.text_box: 0,
            YELLOW.ram.facing: 0,
            YELLOW.ram.event_flags: 0,
            YELLOW.ram.num_bag_items: 0,
        })
        self.steps = 0

    def snapshot(self):
        return dict(self.ram)

    def restore(self, blob):
        self.ram = dict(blob)

    def step(self, action_index):
        from poke.actions import ACTIONS
        self.steps += 1
        button = ACTIONS[action_index]
        x = self.ram[YELLOW.ram.x_coord]
        if button == "right" and x < CORRIDOR - 1:
            self.ram[YELLOW.ram.x_coord] = x + 1
        if button == "left" and x > 0:
            self.ram[YELLOW.ram.x_coord] = x - 1
        if button == "a" and x == CORRIDOR - 1:
            self.ram[YELLOW.ram.event_flags] = 1


def test_search_finds_the_shortest_route_down_the_corridor():
    env = CorridorEnv()
    result = search(
        env,
        goal=lambda: env.ram[YELLOW.ram.x_coord] == CORRIDOR - 1,
        allowed={1},
        buttons=("left", "right"),
        with_facing=False,
    )
    assert result.found
    assert result.path == ["right"] * (CORRIDOR - 1)
    assert result.end == (1, 0, CORRIDOR - 1)


def test_search_reports_failure_rather_than_hanging():
    env = CorridorEnv()
    result = search(
        env,
        goal=lambda: env.ram[YELLOW.ram.x_coord] == 99,
        allowed={1},
        buttons=("left", "right"),
        with_facing=False,
        max_nodes=50,
    )
    assert not result.found
    assert result.path == []
    assert result.nodes <= 50


def test_search_will_not_leave_the_allowed_maps():
    env = CorridorEnv()
    result = search(
        env,
        goal=lambda: False,
        allowed=set(),
        buttons=("right",),
        with_facing=False,
        max_nodes=20,
    )
    assert not result.found
    assert result.maps_seen == {}


def test_a_press_that_flips_a_flag_is_reachable():
    env = CorridorEnv()
    result = search(
        env,
        goal=lambda: env.ram[YELLOW.ram.event_flags] == 1,
        allowed={1},
        buttons=("right", "a"),
        with_facing=False,
    )
    assert result.found
    assert result.path[-1] == "a"


def test_story_key_separates_states_that_differ_only_in_flags():
    env = CorridorEnv()
    before = story_key(env, with_facing=False)
    env.ram[YELLOW.ram.event_flags] = 1
    assert story_key(env, with_facing=False) != before
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_oracle.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'poke.oracle'`

- [ ] **Step 3: Write `poke/oracle.py`**

Port the spike, keeping its two hard-won fixes: settle after a warp before judging a child, and reject a blackout warp as a route. Mark the module training-only at the top.

```python
"""Breadth-first search over the game, with the emulator as the model.

TRAINING ONLY. This snapshots and restores on every expansion, which the
play-time contract bans outright. Its jobs are to produce milestone save
states, to produce demonstration frames for the encoder, and to be the ceiling
a learned policy is measured against -- labelled as using rollback.

Ported from the throwaway spike of 2026-10-21, which reached the Viridian
Forest south gate in about 21 minutes of CPU.
"""

from __future__ import annotations

import collections
import hashlib
import time
from dataclasses import dataclass, field

from poke.actions import ACTIONS
from poke.flags import NUM_EVENT_BYTES, bag_items

# Maps the game warps a blacked-out player to. Arriving at one after a battle
# is not a route, it is a loss, and a search that accepts it will "find" a
# shortcut through fainting.
HOME_OR_CENTER = frozenset({37, 38, 41})

# How many no-ops to spend waiting for the fade-in after a warp.
WARP_SETTLE = 30


@dataclass(frozen=True)
class SearchResult:
    found: bool
    path: list[str] = field(default_factory=list)
    expanded: int = 0
    nodes: int = 0
    emulated_actions: int = 0
    maps_seen: dict[int, int] = field(default_factory=dict)
    seconds: float = 0.0
    end: tuple[int, int, int] | None = None


def _pos(env):
    ram = env.game.ram
    return (env.read_ram(ram.cur_map), env.read_ram(ram.y_coord),
            env.read_ram(ram.x_coord))


def story_key(env, with_facing: bool) -> tuple:
    """Where the player is, plus a digest of everything the story remembers.

    Position alone is not a state: standing on the Mart counter before and
    after the parcel is in the bag are different worlds, and a search keyed on
    position would prune the second one away.
    """
    ram = env.game.ram
    flags = bytes(env.read_ram(ram.event_flags + i) or 0
                  for i in range(NUM_EVENT_BYTES))
    bag = bytes(value for pair in bag_items(env) for value in pair)
    digest = hashlib.blake2b(flags + bag, digest_size=8).digest()
    facing = (env.read_ram(ram.facing) or 0) if with_facing else 0
    return (*_pos(env), facing, digest)


def _has_control(env) -> bool:
    """The rollback probe. Legal here, banned at play time."""
    blob = env.snapshot()
    screens = []
    try:
        for button in ("down", "right"):
            env.restore(blob)
            env.step(ACTIONS.index(button))
            screens.append(env.screen()[:, :, 0].tobytes())
    finally:
        env.restore(blob)
    return screens[0] != screens[1]


def transition(env, button: str, cap: int = 400):
    """One action, then mash A through whatever it started.

    Returns (actions spent, saw a battle), or None if this is not a route:
    either the game never gave control back inside `cap`, or the player blacked
    out and was warped home.
    """
    ram = env.game.ram
    before = _pos(env)
    env.step(ACTIONS.index(button))
    used, saw_battle = 1, False
    if _pos(env)[0] != before[0]:
        # A warp lands mid fade-in and the next input is swallowed. Without
        # this the child state equals its parent and the search discards every
        # door it walks through.
        for _ in range(WARP_SETTLE):
            if _has_control(env):
                break
            env.step(ACTIONS.index("noop"))
            used += 1
    if button != "a" and _pos(env) != before and not env.read_ram(ram.in_battle):
        return used, saw_battle
    while used < cap:
        if env.read_ram(ram.in_battle):
            saw_battle = True
            env.step(ACTIONS.index("a"))
            used += 1
            continue
        if _has_control(env):
            break
        env.step(ACTIONS.index("a"))
        used += 1
    else:
        return None
    if saw_battle and _pos(env)[0] in HOME_OR_CENTER and before[0] not in HOME_OR_CENTER:
        return None
    return used, saw_battle


def search(env, goal, allowed, buttons=("up", "down", "left", "right", "a"),
           with_facing: bool = True, max_nodes: int = 12000,
           on_progress=None) -> SearchResult:
    started = time.time()
    root_blob = env.snapshot()
    root = story_key(env, with_facing)
    parents: dict[tuple, tuple] = {root: (None, None, root_blob)}
    queue = collections.deque([root])
    expanded = emulated = 0
    maps_seen: collections.Counter = collections.Counter()
    found = None

    while queue and len(parents) < max_nodes and found is None:
        node = queue.popleft()
        _, _, blob = parents[node]
        expanded += 1
        for button in buttons:
            env.restore(blob)
            outcome = transition(env, button)
            if outcome is None:
                continue
            emulated += outcome[0]
            here = _pos(env)
            if here[0] not in allowed:
                continue
            child = story_key(env, with_facing)
            if child in parents:
                continue
            parents[child] = (node, button, env.snapshot())
            maps_seen[here[0]] += 1
            if goal():
                found = child
                break
            queue.append(child)
        if on_progress and expanded % 250 == 0:
            on_progress(expanded, len(parents), len(queue), dict(maps_seen))

    path: list[str] = []
    end = None
    if found is not None:
        node = found
        while parents[node][0] is not None:
            parent, button, _ = parents[node]
            path.append(button)
            node = parent
        path.reverse()
        env.restore(parents[found][2])
        end = _pos(env)

    return SearchResult(
        found=found is not None, path=path, expanded=expanded,
        nodes=len(parents), emulated_actions=emulated,
        maps_seen=dict(maps_seen), seconds=round(time.time() - started, 1),
        end=end,
    )
```

- [ ] **Step 4: Run the test**

Run: `uv run pytest tests/test_oracle.py -q`
Expected: PASS.

- [ ] **Step 5: Write `scripts/oracle.py`**

```python
# scripts/oracle.py
"""Run the oracle over Part 2 and write what training needs from it.

TRAINING ONLY: this uses emulator snapshots, which the play-time contract bans.

Writes one save state per milestone into --states-dir and a JSON report with
the path, the node counts and the wall-clock cost of each phase.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from poke.env import PokeEnv
from poke.flags import GOT_POKEDEX, carrying_parcel, event_bit
from poke.objectives import VIRIDIAN_FOREST
from poke.oracle import search

FOREST_GATE = 50
PHASES = ("parcel", "pokedex", "forest")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rom", default="pokeyellow.gbc")
    parser.add_argument("--state", default="states/route1.state",
                        help="where to start; the Route 1 state for stage A")
    parser.add_argument("--states-dir", default="states/oracle")
    parser.add_argument("--out", default="results/oracle.json")
    parser.add_argument("--max-nodes", type=int, default=20000)
    args = parser.parse_args()

    env = PokeEnv(args.rom, args.state)
    env.reset()
    Path(args.states_dir).mkdir(parents=True, exist_ok=True)
    report = {}

    def save(name):
        path = Path(args.states_dir) / f"{name}.state"
        path.write_bytes(env.snapshot())
        return str(path)

    def progress(expanded, nodes, queued, maps):
        print(f"expanded {expanded} nodes {nodes} queue {queued} maps {maps}",
              flush=True)

    phases = [
        ("parcel", lambda: carrying_parcel(env), {12, 1, 42},
         ("up", "down", "left", "right", "a")),
        ("pokedex", lambda: event_bit(env, GOT_POKEDEX) == 1, {42, 1, 12, 0, 40},
         ("up", "down", "left", "right", "a")),
        ("forest", lambda: (env.read_ram(env.game.ram.cur_map) == VIRIDIAN_FOREST),
         {40, 0, 12, 1, 13, FOREST_GATE, VIRIDIAN_FOREST},
         ("up", "down", "left", "right")),
    ]
    for name, goal, allowed, buttons in phases:
        result = search(env, goal, allowed, buttons=buttons,
                        with_facing="a" in buttons, max_nodes=args.max_nodes,
                        on_progress=progress)
        report[name] = {
            "found": result.found, "path": result.path,
            "path_len": len(result.path), "expanded": result.expanded,
            "nodes": result.nodes, "emulated_actions": result.emulated_actions,
            "maps_seen": result.maps_seen, "seconds": result.seconds,
            "end": result.end,
        }
        print(json.dumps({k: v for k, v in report[name].items() if k != "path"}),
              flush=True)
        if not result.found:
            break
        report[name]["state"] = save(name)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Confirm map 51 on the cartridge**

The spike's search stopped at `(50, 7, 4)` because its goal accepted map 50 or 51. The spec flags map 51 as **unconfirmed**. Run the forest phase with the goal above, which accepts 51 only:

Run: `uv run python -m scripts.oracle --state states/route1.state`
Expected: `report["forest"]["found"] is True` and `report["forest"]["end"][0] == 51`.

If it comes back False, the gate building's exit is not where the search can reach it: re-run that phase alone from `states/oracle/pokedex.state` with `--max-nodes 40000` and `allowed` extended with the gate's second map id as printed in `maps_seen`. **Record whichever it is in `results/oracle.json` and update the spec's A3 row from "unconfirmed" to the measured value.** This is the one open fact in the spec's win condition; do not leave it open.

- [ ] **Step 7: Verify `--help` works without a ROM**

Run: `uv run python -m scripts.oracle --help`
Expected: usage text, exit 0, no PyBoy instantiation. (CI has no ROM.)

- [ ] **Step 8: Commit**

```bash
git add poke/oracle.py scripts/oracle.py tests/test_oracle.py results/oracle.json docs/superpowers/specs/2026-10-21-part2-world-model-rl-design.md
git commit -m "The oracle moves into the repo, and map 51 stops being a guess

BFS over (map, y, x, facing, story digest) with the emulator as the model.
Training only: it snapshots on every expansion, which play time bans.

Two fixes carried over from the spike. A warp lands mid fade-in, so the search
settles before judging the child, or every door looks like a dead end. And a
blackout warp home is rejected as a route, or fainting becomes a shortcut.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: The play-time contract, enforced

**Files:**
- Create: `poke/contract.py`
- Test: `tests/test_contract.py`

**Why:** the spec's anti-goals include "any play-time read of coordinates, map id, waypoints or walls, and any rollback". An anti-goal that is only a promise gets broken by the third refactor. This makes it raise.

**Interfaces:**
- Consumes: `poke.env.PokeEnv`, `poke.handlers.HandlerState`, `poke.objectives`.
- Produces: `poke.contract.PlayTimeEnv(env)` — a proxy that forwards everything, raises `ContractViolation` on `read_ram` of `cur_map`/`y_coord`/`x_coord` and on `snapshot`/`restore`; `poke.contract.ContractViolation(Exception)`; `poke.contract.Run` with `record(action_index, controller) -> None`, `log: list[dict]`, `by_controller: dict[str, int]`, `learned_share: float`, `tiles_seen: set`, `new_tiles_this_objective: int`.

**Design note:** `Run` is the one place allowed to read position, because it is the scorer and the reward, not the policy. It takes the **unwrapped** env. The policy only ever sees `PlayTimeEnv`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_contract.py
import pytest

from poke.actions import ACTIONS
from poke.contract import ContractViolation, PlayTimeEnv, Run
from poke.games import YELLOW
from poke.objectives import get
from tests.conftest import FakeEnv

A = ACTIONS.index("a")
UP = ACTIONS.index("up")


def base_env(**ram):
    return FakeEnv(YELLOW, {
        YELLOW.ram.cur_map: 0, YELLOW.ram.y_coord: 5, YELLOW.ram.x_coord: 5,
        YELLOW.ram.in_battle: 0, YELLOW.ram.text_box: 0,
        YELLOW.ram.party_count: 1, YELLOW.ram.event_flags: 0,
        YELLOW.ram.num_bag_items: 0, **ram,
    })


def test_the_policys_env_refuses_to_report_position():
    guarded = PlayTimeEnv(base_env())
    for addr in (YELLOW.ram.cur_map, YELLOW.ram.y_coord, YELLOW.ram.x_coord):
        with pytest.raises(ContractViolation, match="position"):
            guarded.read_ram(addr)


def test_the_policys_env_refuses_to_roll_back():
    guarded = PlayTimeEnv(base_env())
    with pytest.raises(ContractViolation, match="rollback"):
        guarded.snapshot()
    with pytest.raises(ContractViolation, match="rollback"):
        guarded.restore(b"")


def test_flags_and_the_bag_still_read_fine():
    guarded = PlayTimeEnv(base_env(**{YELLOW.ram.event_flags: 0b100000}))
    assert guarded.read_ram(YELLOW.ram.event_flags) == 0b100000


def test_learned_share_counts_only_the_policys_actions():
    run = Run(base_env())
    run.record(UP, "policy")
    run.record(A, "text")
    run.record(A, "battle")
    run.record(UP, "policy")
    assert run.by_controller == {"policy": 2, "text": 1, "battle": 1}
    assert run.learned_share == 0.5


def test_no_control_turns_count_as_neither():
    """Walking cutscenes and the swallowed input after a warp are nobody's."""
    run = Run(base_env())
    run.record(UP, "policy")
    run.record(ACTIONS.index("noop"), "no_control")
    assert run.by_controller["no_control"] == 1
    assert run.learned_share == 1.0


def test_two_tile_step_counts_as_one_action_and_two_new_tiles():
    """One 60-frame action sometimes walks two tiles, so consecutive positions
    are not always adjacent. Tile counting must not assume they are."""
    env = base_env()
    run = Run(env)
    run.record(UP, "policy")
    env.write(YELLOW.ram.y_coord, 3)
    run.record(UP, "policy")
    assert len(run.log) == 2
    assert run.tiles_seen == {(0, 5, 5), (0, 3, 5)}


def test_blackout_warp_is_logged_and_does_not_reset_the_objective():
    """Blacking out is not special-cased: the player reappears at home and the
    run carries on paying for it in actions."""
    env = base_env()
    run = Run(env)
    run.record(UP, "policy")
    env.write(YELLOW.ram.cur_map, 37)   # warped home
    env.write(YELLOW.ram.y_coord, 1)
    env.write(YELLOW.ram.x_coord, 2)
    run.record(UP, "policy")
    assert run.log[-1]["map"] == 37
    assert (37, 1, 2) in run.tiles_seen
    assert run.objective_key == get("trigger_oak").key


def test_new_tiles_reset_when_the_objective_changes():
    """The visited set is per objective, which is what makes the backtrack to
    Oak pay a second time."""
    env = base_env()
    run = Run(env)
    run.record(UP, "policy")
    # OAK_APPEARED_IN_PALLET is bit 39: byte 4, bit 7.
    env.write(YELLOW.ram.event_flags + 39 // 8, 1 << (39 % 8))
    run.record(UP, "policy")
    assert run.new_tiles_this_objective == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_contract.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'poke.contract'`

- [ ] **Step 3: Write `poke/contract.py`**

```python
"""The play-time contract, as code that raises.

At training time anything goes: RAM rewards, save-state resets, oracle routes.
At play time the policy sees pixels, the frozen encoder's embeddings and which
objective it is on, and nothing else. In particular it may not read the map id
or the coordinates, and the run may not roll back.

Those are anti-goals in the design document. Anti-goals that are only written
down get broken quietly, so `PlayTimeEnv` turns them into exceptions. The
scorer -- `Run` -- keeps the unguarded env, because judging a run is allowed to
know where the player is.
"""

from __future__ import annotations

from poke.actions import ACTIONS
from poke.objectives import current


class ContractViolation(Exception):
    pass


class PlayTimeEnv:
    """Everything `PokeEnv` does, minus what the contract forbids."""

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        if name in ("snapshot", "restore"):
            raise ContractViolation(
                f"{name}() is rollback, which play time bans: the run is one "
                "continuous episode. Use it in training code only."
            )
        return getattr(self._env, name)

    def read_ram(self, addr):
        ram = self._env.game.ram
        if addr in (ram.cur_map, ram.y_coord, ram.x_coord):
            raise ContractViolation(
                f"{addr:#06x} is position. The policy navigates from pixels; "
                "map id and coordinates are for scoring, on the unguarded env."
            )
        return self._env.read_ram(addr)


class Run:
    """One continuous episode's log, controller tally and tile coverage.

    Holds the unguarded env on purpose: this is the scorer.
    """

    #: Controllers whose actions are neither learned nor handler work.
    UNOWNED = ("no_control",)

    def __init__(self, env):
        self.env = env
        self.log: list[dict] = []
        self.by_controller: dict[str, int] = {}
        self.tiles_seen: set[tuple] = set()
        self.objective_key = current(env).key
        self.new_tiles_this_objective = 0

    def _tile(self):
        ram = self.env.game.ram
        return (self.env.read_ram(ram.cur_map), self.env.read_ram(ram.y_coord),
                self.env.read_ram(ram.x_coord))

    def record(self, action_index: int, controller: str) -> None:
        """Called once per emulator action, after it has been taken."""
        objective = current(self.env)
        if objective.key != self.objective_key:
            # A new objective re-opens the whole map: the walk back to Oak is
            # new ground again, which is what makes the backtrack pay.
            self.objective_key = objective.key
            self.tiles_seen = set()
            self.new_tiles_this_objective = 0
        tile = self._tile()
        if tile not in self.tiles_seen:
            self.tiles_seen.add(tile)
            self.new_tiles_this_objective += 1
        self.by_controller[controller] = self.by_controller.get(controller, 0) + 1
        self.log.append({
            "n": len(self.log) + 1,
            "objective": objective.key,
            "controller": controller,
            "action": ACTIONS[action_index],
            "map": tile[0], "y": tile[1], "x": tile[2],
        })

    @property
    def learned_share(self) -> float:
        """Policy actions over policy plus handler actions.

        Turns nobody had -- walking cutscenes, the input the fade-in ate --
        are excluded from both sides rather than counted against the policy.
        """
        owned = {k: v for k, v in self.by_controller.items() if k not in self.UNOWNED}
        total = sum(owned.values())
        return 0.0 if total == 0 else owned.get("policy", 0) / total
```

- [ ] **Step 4: Run the test**

Run: `uv run pytest tests/test_contract.py -q && uv run pytest -q`
Expected: PASS, suite green.

- [ ] **Step 5: Commit**

```bash
git add poke/contract.py tests/test_contract.py
git commit -m "Make the play-time contract raise instead of promise

PlayTimeEnv forwards everything except the three position addresses and
snapshot/restore, which throw ContractViolation. The policy gets that object;
the scorer keeps the real one, because judging a run may know where the player
stood.

Run tallies the learned share, with no_control turns excluded from both sides,
and resets the visited-tile set when the objective changes.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: A contract-compliant runner and the cheap baselines

**Files:**
- Create: `scripts/play2.py`
- Test: `tests/test_play2.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `scripts.play2.policy_for(name, rng) -> Callable[[object, list[float]], int]` for `name` in `{"random", "noop", "up", "down", "left", "right", "a", "b", "oracle"}`; `scripts.play2.play(env, policy, budget, run=None) -> dict` returning `{"objectives": {key: actions_or_None}, "learned_share": float, "actions": int, "by_controller": dict}`.

The runner is the spec's eval protocol minus the trained policy: one continuous episode, a fixed action budget counting handler actions, no rollback.

**One deliberate difference from the spec.** The spec puts the handlers "inside the env wrapper", so one policy action expands into several handler actions before the next observation. `play` instead gives each handler action its own turn of the loop. The accounting is identical — every emulator action is logged with its controller and charged to the budget either way — and the flat loop is what makes the contract testable, because the policy is called at exactly one place. When the RL plan adds a vectorised env, the expansion moves inside it and this loop becomes the single-worker path; the per-action log is the interface both share.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_play2.py
import random

from poke.actions import ACTIONS
from poke.contract import ContractViolation
from poke.games import YELLOW
from scripts.play2 import play, policy_for
from tests.test_oracle import CorridorEnv


def test_constant_policy_presses_only_its_button():
    policy = policy_for("up", random.Random(0))
    assert all(policy(None, [0.0] * 5) == ACTIONS.index("up") for _ in range(10))


def test_random_policy_is_seeded_and_reproducible():
    first = [policy_for("random", random.Random(7))(None, [0.0] * 5) for _ in range(20)]
    second = [policy_for("random", random.Random(7))(None, [0.0] * 5) for _ in range(20)]
    assert first == second


def test_play_stops_at_the_budget():
    env = CorridorEnv()
    result = play(env, policy_for("left", random.Random(0)), budget=12)
    assert result["actions"] == 12


def test_play_hands_the_policy_a_guarded_env():
    """A policy that peeks at the map must blow up, not quietly win."""
    def cheater(env, objective):
        env.read_ram(YELLOW.ram.cur_map)
        return ACTIONS.index("right")

    env = CorridorEnv()
    try:
        play(env, cheater, budget=4)
    except ContractViolation as error:
        assert "position" in str(error)
    else:
        raise AssertionError("the runner handed the policy an unguarded env")


def test_handler_actions_count_against_the_budget():
    env = CorridorEnv()
    env.ram[YELLOW.ram.text_box] = 1          # a box is open the whole time
    result = play(env, policy_for("right", random.Random(0)), budget=10)
    assert result["actions"] == 10
    assert result["by_controller"].get("policy", 0) == 0
    assert result["learned_share"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_play2.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.play2'`

- [ ] **Step 3: Write `scripts/play2.py`**

```python
# scripts/play2.py
"""Play Part 2 under the contract: one continuous run, no rollback.

The policy argument is deliberately dumb here -- random, a constant button, or
a replay of the oracle's path. The trained policy lands in this same slot, so
whatever this file measures about the baselines it will measure about it.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from poke.actions import ACTIONS, n_actions
from poke.contract import PlayTimeEnv, Run
from poke.env import PokeEnv
from poke.handlers import HandlerState, choose
from poke.objectives import CHAIN, one_hot

POLICIES = ("random", "noop", "up", "down", "left", "right", "a", "b", "oracle")


def policy_for(name: str, rng: random.Random, path=None):
    if name == "random":
        return lambda env, objective: rng.randrange(n_actions())
    if name == "oracle":
        steps = list(path or [])
        def replay(env, objective, _steps=steps):
            return ACTIONS.index(_steps.pop(0)) if _steps else ACTIONS.index("noop")
        return replay
    index = ACTIONS.index(name)
    return lambda env, objective: index


def play(env, policy, budget: int, run: Run | None = None) -> dict:
    """One episode. Handlers take the turns they own; the policy takes the rest.

    Handler actions count against the budget, as the spec requires: a run that
    spends its whole budget in menus has not navigated anywhere.
    """
    run = run or Run(env)
    guarded = PlayTimeEnv(env)
    state = HandlerState()
    reached: dict[str, int | None] = {objective.key: None for objective in CHAIN}

    for _ in range(budget):
        before_map = env.read_ram(env.game.ram.cur_map)
        claimed = choose(env, state)
        if claimed is None:
            action, controller = policy(guarded, one_hot(env)), "policy"
        else:
            action, controller = claimed
        env.step(action)
        run.record(action, controller)
        state.note(action, env.read_ram(env.game.ram.cur_map) != before_map)
        for objective in CHAIN:
            if reached[objective.key] is None and objective.reached(env):
                reached[objective.key] = len(run.log)

    return {
        "objectives": reached,
        "actions": len(run.log),
        "by_controller": dict(run.by_controller),
        "learned_share": run.learned_share,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rom", default="pokeyellow.gbc")
    parser.add_argument("--state", default="states/route1.state")
    parser.add_argument("--policy", default="random", choices=POLICIES)
    parser.add_argument("--oracle-report", default="results/oracle.json",
                        help="where --policy oracle reads its path from")
    parser.add_argument("--budget", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    path = None
    if args.policy == "oracle":
        report = json.loads(Path(args.oracle_report).read_text())
        path = [button for phase in report.values() for button in phase.get("path", [])]

    env = PokeEnv(args.rom, args.state)
    env.reset()
    env.settle()
    result = play(env, policy_for(args.policy, random.Random(args.seed), path),
                  budget=args.budget)
    result["policy"] = args.policy
    result["seed"] = args.seed
    print(json.dumps(result, indent=1))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test**

Run: `uv run pytest tests/test_play2.py -q && uv run pytest -q`
Expected: PASS, suite green.

- [ ] **Step 5: Run the cheap baselines on the cartridge**

Run, once per policy:

```bash
for p in random noop up down left right a b; do
  uv run python -m scripts.play2 --policy $p --budget 4000 \
    --out results/part2_baseline_$p.json
done
uv run python -m scripts.play2 --policy oracle --budget 4000 \
  --out results/part2_oracle.json
```

Expected: the constant buttons and random reach **no** Part 2 objective, and the oracle replay reaches `enter_forest`. **If any constant button clears an objective, that objective is broken** — the same check `scripts/eval_baselines.py` already makes for Part 1 — and the fix is the objective, not the baseline.

- [ ] **Step 6: Commit**

```bash
git add scripts/play2.py tests/test_play2.py results/part2_baseline_*.json results/part2_oracle.json
git commit -m "Run Part 2 under the contract, with the cheap baselines first

One continuous episode, handler actions counted against the budget, the policy
handed a PlayTimeEnv so a position read raises. The trained policy goes in the
same slot, so the baselines and the policy are measured by the same code.

A table of zeros for the constant buttons is the point: it says the objectives
are not reachable by accident.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Measure multi-process PyBoy throughput

**Files:**
- Create: `scripts/bench_workers.py`
- Test: `tests/test_bench_workers.py`

**Why:** the spec says multi-process PyBoy throughput "is **unmeasured**, so measure it first", and the RL plan's training budget is fixed from this number. It is the last task here and the first input there.

**Interfaces:**
- Produces: `scripts.bench_workers.summarise(per_worker_rates) -> dict` with keys `workers`, `total_steps_per_s`, `per_worker_steps_per_s`, `efficiency`; the script writes `results/worker_throughput.json`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bench_workers.py
import pytest

from scripts.bench_workers import summarise


def test_perfect_scaling_is_efficiency_one():
    out = summarise([10.0, 10.0, 10.0, 10.0], single=10.0)
    assert out["workers"] == 4
    assert out["total_steps_per_s"] == 40.0
    assert out["efficiency"] == 1.0


def test_contention_shows_up_as_efficiency_below_one():
    out = summarise([5.0, 5.0], single=10.0)
    assert out["total_steps_per_s"] == 10.0
    assert out["efficiency"] == 0.5


def test_no_workers_is_an_error_not_a_zero_division():
    with pytest.raises(ValueError, match="at least one worker"):
        summarise([], single=10.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_bench_workers.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.bench_workers'`

- [ ] **Step 3: Write `scripts/bench_workers.py`**

```python
# scripts/bench_workers.py
"""How many emulator steps per second does this machine do, in parallel?

The encoder runs at ~1,758 frames/s and training at ~403 frames/s, both
measured. The emulator's parallel rate is not, and it is what decides how long
an RL run takes. So: N processes, each stepping its own PokeEnv with random
actions, and the honest total.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from multiprocessing import Pool
from pathlib import Path

from poke.actions import n_actions


def _worker(job):
    rom, state, steps, seed = job
    from poke.env import PokeEnv           # imported here: one PyBoy per process
    rng = random.Random(seed)
    env = PokeEnv(rom, state)
    env.reset()
    started = time.time()
    for _ in range(steps):
        env.step(rng.randrange(n_actions()))
    elapsed = time.time() - started
    env.close()
    return steps / elapsed


def summarise(per_worker_rates, single: float) -> dict:
    if not per_worker_rates:
        raise ValueError("need at least one worker to summarise")
    total = sum(per_worker_rates)
    return {
        "workers": len(per_worker_rates),
        "total_steps_per_s": round(total, 2),
        "per_worker_steps_per_s": round(total / len(per_worker_rates), 2),
        "single_worker_steps_per_s": round(single, 2),
        "efficiency": round(total / (single * len(per_worker_rates)), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rom", default="pokeyellow.gbc")
    parser.add_argument("--state", default="states/bedroom.state")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--out", default="results/worker_throughput.json")
    args = parser.parse_args()

    report = {}
    single = _worker((args.rom, args.state, args.steps, 0))
    for count in args.workers:
        jobs = [(args.rom, args.state, args.steps, seed) for seed in range(count)]
        with Pool(count) as pool:
            rates = pool.map(_worker, jobs)
        report[str(count)] = summarise(rates, single)
        print(json.dumps(report[str(count)]), flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    best = max(report.values(), key=lambda row: row["total_steps_per_s"])
    print(f"best: {best['workers']} workers, "
          f"{best['total_steps_per_s']} steps/s, efficiency {best['efficiency']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test, then the benchmark**

Run: `uv run pytest tests/test_bench_workers.py -q`
Expected: PASS.

Run: `uv run python -m scripts.bench_workers --steps 300 --workers 1 2 4 8 12`
Expected: a table. The number that matters for the RL plan is `total_steps_per_s` at the best worker count. Note it — one PPO environment step is one emulator action, so training wall-clock is `total_actions / total_steps_per_s` plus the policy's own forward passes.

- [ ] **Step 5: Commit**

```bash
git add scripts/bench_workers.py tests/test_bench_workers.py results/worker_throughput.json
git commit -m "Measure parallel emulator throughput before budgeting any RL

Encoder inference and world-model training were measured; the emulator in
parallel was not, and it is the term that decides how long a PPO run takes.
N processes, each with its own PokeEnv, random actions, and the efficiency
against a single worker.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## After this plan

With Tasks 1-8 merged the repo can run Part 2 end to end under the contract, with the oracle as the ceiling and the constant-button table as the floor, and it knows how fast it can emulate in parallel. The RL plan is then written against three measured numbers: the oracle's action count per stage (the budget is 3× it), the parallel step rate, and the control probe's agreement. Its tasks are, in order: dataset refresh over the Part 2 maps and encoder retrain; the actor-critic and a hand-rolled PPO; the Go-Explore archive for resets; the intrinsic bonus, chosen by a small prediction-error-vs-disagreement comparison; the three arms; and the eval protocol with its GIFs.
