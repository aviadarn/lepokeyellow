"""The training reward. Never a policy input.

From the design: +1 per objective completed, +0.01 per new (map, y, x) tile
within the current objective, and the visited set resets when the objective
changes -- which is what makes the walk back to Oak pay a second time. No
level, heal or battle terms: prior art shows every one of those got hacked.

Two things this file is careful about, both learned from the cartridge rather
than from the design. An objective is credited at most once per episode, held
in a set rather than an index, because `poke.objectives.current` reads flags
and the bag and can report an earlier objective than one already finished --
the bag empties the moment Oak takes the parcel. And objectives already true
when the episode starts are never credited: an archive state that begins after
the Mart would otherwise pay +1 for a save state rather than for a walk.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from poke.objectives import CHAIN, current

#: What one milestone is worth.
OBJECTIVE_REWARD = 1.0
#: What one previously unvisited tile is worth, inside the current objective.
TILE_REWARD = 0.01


@dataclass
class RewardState:
    """Per-episode reward memory. Reset with the episode, never shared."""

    objective_key: str
    tiles: set[tuple] = field(default_factory=set)
    credited: set[str] = field(default_factory=set)

    @classmethod
    def start(cls, env) -> "RewardState":
        return cls(
            objective_key=current(env).key,
            credited={o.key for o in CHAIN if o.reached(env)},
        )


def step_reward(env, state: RewardState) -> tuple[float, list[str]]:
    """Reward for the transition just taken, and the objectives it completed.

    Call once per emulator action, after the action, including handler actions:
    the parcel is collected by a handler's A press, and an objective that only
    a policy turn could pay for would never be paid.
    """
    reward, completed = 0.0, []
    for objective in CHAIN:
        if objective.key not in state.credited and objective.reached(env):
            state.credited.add(objective.key)
            completed.append(objective.key)
            reward += OBJECTIVE_REWARD

    key = current(env).key
    if key != state.objective_key:
        state.objective_key = key
        state.tiles = set()

    ram = env.game.ram
    tile = (env.read_ram(ram.cur_map), env.read_ram(ram.y_coord),
            env.read_ram(ram.x_coord))
    if tile not in state.tiles:
        state.tiles.add(tile)
        reward += TILE_REWARD

    return reward, completed
