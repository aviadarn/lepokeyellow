"""An ordered chain of subgoals, and where each one's goal image comes from.

A reward-free world model cannot decide what to do next, so somebody has to
supply the curriculum. Upstream supplies exactly one goal and plans to it once,
open-loop, over 14 actions. Playing the game means a chain, and replanning
against the real screen between steps.

The goal *images* need no extra collection: `scripts/bootstrap.py` already
writes a verified save state for each point in the story, and the screen of
that state is precisely the frame the planner should be steering towards. So a
milestone is a RAM predicate (did we get there) plus the state file whose
screen says what getting there looks like.

The chain below is the first scoped target: through the lab and out of Pallet.
It deliberately contains exactly one battle -- the rival fight that Oak's
handover forces -- because the planner does not steer battles and the handoff
has to be exercised rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from poke.maps import OAKS_LAB, PALLET_TOWN, REDS_HOUSE_1F, REDS_HOUSE_2F
from poke.ram import TRAINER_BATTLE, battle_kind, cur_map, has_pokemon

# Maps that still count as "has not left Pallet yet".
PALLET_AREA = frozenset({PALLET_TOWN, REDS_HOUSE_1F, REDS_HOUSE_2F, OAKS_LAB})


@dataclass(frozen=True)
class Milestone:
    key: str
    description: str
    reached: Callable[[object, dict], bool]
    goal_state: str | None
    horizon: int

    def is_plannable(self) -> bool:
        """Without a goal image there is nothing for CEM to steer towards."""
        return self.goal_state is not None


def _has_pokemon(env, history):
    return has_pokemon(env)


def _rival_beaten(env, history):
    """A trainer battle was seen, it is over, and the party survived.

    Checking "not in battle" alone is true before the fight as well, so the
    history has to record that the fight happened.
    """
    return (
        history.get("saw_trainer_battle", False)
        and battle_kind(env) != TRAINER_BATTLE
        and has_pokemon(env)
    )


def _left_lab(env, history):
    return has_pokemon(env) and cur_map(env) not in (None, OAKS_LAB)


def _left_pallet(env, history):
    here = cur_map(env)
    return here is not None and here not in PALLET_AREA


CHAIN: tuple[Milestone, ...] = (
    Milestone(
        key="pikachu",
        description="hold a Pokemon (Oak hands over Pikachu)",
        reached=_has_pokemon,
        goal_state="states/pikachu.state",
        horizon=24,
    ),
    Milestone(
        key="rival_beaten",
        description="survive the forced rival battle in the lab",
        reached=_rival_beaten,
        # Scripted A-mashing clears this, so no goal image is needed: 43 A
        # presses won it on the cartridge with Pikachu intact.
        goal_state=None,
        horizon=24,
    ),
    Milestone(
        key="left_lab",
        description="get back out of Oak's lab",
        reached=_left_lab,
        goal_state="states/pallet.state",
        horizon=24,
    ),
    Milestone(
        key="left_pallet",
        description="leave Pallet Town",
        reached=_left_pallet,
        goal_state=None,
        horizon=24,
    ),
)


def chain_from(key: str | None = None) -> tuple[Milestone, ...]:
    if key is None:
        return CHAIN
    keys = [m.key for m in CHAIN]
    if key not in keys:
        raise ValueError(f"unknown milestone {key!r}; known: {', '.join(keys)}")
    return CHAIN[keys.index(key):]


def get(key: str) -> Milestone:
    for milestone in CHAIN:
        if milestone.key == key:
            return milestone
    raise ValueError(f"unknown milestone {key!r}")
