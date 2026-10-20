"""Task definitions.

Upstream hard-codes one task — "take one of the three balls in Oak's lab" —
into the collector, the planner and the evaluator, by naming episodes after
the three starters. That task does not exist in Yellow: Oak's lab holds a
single ball, the rival takes it, and Oak then hands the player Pikachu
through dialogue. Nothing the player does changes which Pokemon they get.

A task is therefore made explicit here: what counts as success, read from
RAM, plus the horizon and the start state it is defined against. The planner
and evaluator take a task name, so Yellow does not need a forked copy of
either.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from poke.ram import cur_map, has_pokemon

# State carried from reset so a goal can be relative to where the episode began
# (leaving a room means "a different map than the one we started in").
Context = dict


@dataclass(frozen=True)
class Task:
    key: str
    games: tuple[str, ...]
    description: str
    goal: Callable[[object, Context], bool]
    start_context: Callable[[object], Context]
    horizon: int
    state: str
    route: str | None = None

    def supports(self, game_key: str) -> bool:
        return game_key in self.games


def _no_context(env) -> Context:
    return {}


def _map_context(env) -> Context:
    return {"start_map": cur_map(env)}


def _left_start_map(env, context: Context) -> bool:
    here = cur_map(env)
    start = context.get("start_map")
    return here is not None and start is not None and here != start


TASKS: dict[str, Task] = {
    # The upstream task, kept so Red numbers stay comparable to the blog post.
    "red_starter": Task(
        key="red_starter",
        games=("red",),
        description=(
            "From in front of the middle ball in Oak's lab, acquire any starter. "
            "Reproduces stmonty/lePokeRed. Note that 12 A presses already solve "
            "it, which is why eval_baselines exists."
        ),
        goal=lambda env, _: has_pokemon(env),
        start_context=_no_context,
        horizon=14,
        state="states/red_pokeballs.state",
        route="routes/red_starter.json",
    ),
    # Yellow's analogue, defined from a NEW GAME rather than from inside the lab.
    #
    # The obvious framing -- start in Oak's lab, acquire a Pokemon -- is not a
    # hard task, it is an impossible one. The lab is enterable long before the
    # story will hand anything over: Oak has to stop the player north of Pallet
    # first. Bootstrapping to map 40 and running the baselines there scored 0/1
    # for all seven constant buttons and 0/50 for random, which reads like a
    # brutally hard task and is really a start state the goal cannot be reached
    # from. Measured, not assumed -- see results/.
    #
    # From the bootstrap state a random walk does reach it, so the floor is
    # real and worth beating. The horizon has to be large to match: random
    # needed 525-2,600 actions across seeds.
    "yellow_pikachu": Task(
        key="yellow_pikachu",
        games=("yellow",),
        description=(
            "From a new game (controllable, in the bedroom, empty party), end "
            "up holding a Pokemon. Covers the walk out of the house, the Oak "
            "encounter north of Pallet, and the lab scene."
        ),
        goal=lambda env, _: has_pokemon(env),
        start_context=_no_context,
        horizon=600,
        state="states/bedroom.state",
        route="routes/yellow_pikachu.json",
    ),
    # Version-independent navigation task. Success is a map transition, so it is
    # defined from any interior start state and in any Gen-1 game, which makes
    # it the one task that supports a cross-version transfer table.
    "leave_room": Task(
        key="leave_room",
        games=("red", "yellow"),
        description=(
            "Reach any map other than the starting one. Pure navigation: A "
            "alone cannot solve it. Defined from any interior save state, so it "
            "gives a family of start states at different distances from a door."
        ),
        goal=_left_start_map,
        start_context=_map_context,
        horizon=24,
        state="states/oaks_lab.state",
        route=None,
    ),
}


def get(key: str, game_key: str | None = None) -> Task:
    if key not in TASKS:
        raise ValueError(f"unknown task {key!r}; known: {', '.join(sorted(TASKS))}")
    task = TASKS[key]
    if game_key is not None and not task.supports(game_key):
        raise ValueError(
            f"task {key!r} is defined for {', '.join(task.games)}, not {game_key!r}"
        )
    return task


def for_game(game_key: str) -> list[Task]:
    return [task for task in TASKS.values() if task.supports(game_key)]
