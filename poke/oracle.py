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
    """The rollback probe: two buttons from one state, two different screens.

    Exact, expensive, and legal here. `poke.control` is the play-time
    replacement, and it is a measured approximation precisely because this one
    cannot be used there.
    """
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
