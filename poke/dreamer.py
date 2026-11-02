"""Trajectories in the schema of PokeDreamer, for Yellow.

PokeDreamer (huggingface.co/datasets/xxxTEMPESTxxx/PokeDreamer, v2, MIT) is
200 k Pokemon Red transitions from PyBoy for training Dreamer-style world
models: 10 k-step .npz chunks, 24 emulator ticks per action, RGB frames and a
handful of WRAM probes. This module writes the same arrays, names, dtypes and
action indices, so a loader written for it reads these chunks unchanged.

Differences, all additive:
  * Yellow, in GBC colour (PokeDreamer's Red frames are grey);
  * `rewards` is a real signal, with its components kept beside it, where
    PokeDreamer ships a placeholder 0;
  * extra arrays: party_count, levels, event-flag count, the policy that chose
    each action and the start state of each episode.

Step semantics: obs[t] and every probe [t] are read BEFORE actions[t] is
pressed; rewards[t] is what actions[t] earned. episode_starts[t] marks the
first step of an episode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from poke.flags import NUM_EVENT_BYTES
from poke.ram import (badge_count, coords, cur_map, dialog_open, facing,
                      in_battle, party_count, party_hp, party_levels)

#: PokeDreamer's action indices. None is "pass": no button.
ACTIONS = ("down", "left", "right", "up", "a", "b", "start", None)
PASS = 7
TICKS_PER_ACTION = 24
#: The PWhiddy convention PokeDreamer's collection inherits: press, hold for
#: 8 ticks, release, let the rest of the 24 run.
HOLD_TICKS = 8
CHUNK_STEPS = 10_000

#: Name -> (dtype, per-step shape). The first block is PokeDreamer's.
SCHEMA = {
    "obs": (np.uint8, (144, 160, 3)),
    "actions": (np.uint8, ()),
    "rewards": (np.float32, ()),
    "episode_starts": (np.bool_, ()),
    "map_ids": (np.uint8, ()),
    "xs": (np.uint8, ()),
    "ys": (np.uint8, ()),
    "facings": (np.uint8, ()),
    "in_battles": (np.bool_, ()),
    "dialog_opens": (np.bool_, ()),
    "badges": (np.uint8, ()),
    "party_hps": (np.int16, (6,)),
    "party_max_hps": (np.int16, (6,)),
    # Yellow additions.
    "party_counts": (np.uint8, ()),
    "party_levels": (np.uint8, (6,)),
    "event_flag_counts": (np.uint16, ()),
    "reward_explore": (np.float32, ()),
    "reward_level": (np.float32, ()),
    "reward_badge": (np.float32, ()),
    "reward_event": (np.float32, ()),
    "policy_ids": (np.uint8, ()),
    "state_ids": (np.uint16, ()),
}
POKEDREAMER_KEYS = ("obs", "actions", "rewards", "episode_starts", "map_ids", "xs",
                    "ys", "facings", "in_battles", "dialog_opens", "badges",
                    "party_hps", "party_max_hps")

#: rewards = sum(weight * component). Components are stored so a user can
#: re-weight without re-collecting.
REWARD_WEIGHTS = {"explore": 0.01, "level": 1.0, "badge": 5.0, "event": 0.5}


def act(env, action: int) -> None:
    """One PokeDreamer step: hold HOLD_TICKS, release, finish TICKS_PER_ACTION.

    Only the last tick renders: the observation is the screen after the
    action, and rendering the other 23 is wasted work.
    """
    button = ACTIONS[action]
    pyboy = env.pyboy
    if button:
        pyboy.button_press(button)
    pyboy.tick(HOLD_TICKS, render=False)
    if button:
        pyboy.button_release(button)
    pyboy.tick(TICKS_PER_ACTION - HOLD_TICKS - 1, render=False)
    pyboy.tick(1, render=True)


def event_flag_count(env) -> int:
    start = env.game.ram.event_flags
    raw = env.read_ram_range(start, start + NUM_EVENT_BYTES)
    return int(np.unpackbits(np.frombuffer(raw, dtype=np.uint8)).sum())


def probe(env) -> dict:
    """Every per-step field that comes from the emulator, read now."""
    y, x = coords(env)
    hp, max_hp = party_hp(env)
    return {
        "obs": env.screen(),
        "map_ids": cur_map(env) or 0,
        "xs": x or 0,
        "ys": y or 0,
        "facings": facing(env),
        "in_battles": in_battle(env),
        "dialog_opens": dialog_open(env),
        "badges": badge_count(env),
        "party_hps": hp,
        "party_max_hps": max_hp,
        "party_counts": party_count(env),
        "party_levels": party_levels(env),
        "event_flag_counts": event_flag_count(env),
    }


@dataclass
class RewardTracker:
    """Per-episode reward bookkeeping: what is new since the episode began."""

    tiles: set = field(default_factory=set)
    levels: int = 0
    badges: int = 0
    events: int = 0

    def start(self, fields: dict) -> None:
        self.tiles = {(fields["map_ids"], fields["ys"], fields["xs"])}
        self.levels = int(fields["party_levels"].sum())
        self.badges = int(fields["badges"])
        self.events = int(fields["event_flag_counts"])

    def step(self, after: dict) -> dict[str, float]:
        """Components earned by the transition that ended in `after`."""
        tile = (after["map_ids"], after["ys"], after["xs"])
        explore = float(tile not in self.tiles)
        self.tiles.add(tile)
        levels = int(after["party_levels"].sum())
        badges = int(after["badges"])
        events = int(after["event_flag_counts"])
        # Only gains count. A level "drop" is a party change, not a penalty,
        # and a flag cleared by a script is not progress undone.
        out = {"explore": explore,
               "level": float(max(0, levels - self.levels)),
               "badge": float(max(0, badges - self.badges)),
               "event": float(max(0, events - self.events))}
        self.levels = max(self.levels, levels)
        self.badges = max(self.badges, badges)
        self.events = max(self.events, events)
        return out


def total_reward(components: dict[str, float]) -> float:
    return sum(REWARD_WEIGHTS[k] * v for k, v in components.items())


class EpisodeBuffer:
    """Steps of one episode, column-wise, in SCHEMA dtypes."""

    def __init__(self):
        self.columns: dict[str, list] = {key: [] for key in SCHEMA}

    def __len__(self) -> int:
        return len(self.columns["actions"])

    def add(self, fields: dict, action: int, components: dict[str, float],
            policy_id: int, state_id: int, start: bool) -> None:
        row = dict(fields)
        row.update(actions=action, rewards=total_reward(components),
                   episode_starts=start, policy_ids=policy_id, state_ids=state_id,
                   **{f"reward_{k}": v for k, v in components.items()})
        for key in SCHEMA:
            self.columns[key].append(row[key])

    def arrays(self) -> dict[str, np.ndarray]:
        return {key: np.asarray(values, dtype=SCHEMA[key][0]).reshape(
                    (len(values),) + SCHEMA[key][1])
                for key, values in self.columns.items()}


def check(arrays: dict[str, np.ndarray]) -> None:
    """Raise if `arrays` does not match SCHEMA (names, dtypes, shapes, lengths)."""
    missing = set(SCHEMA) - set(arrays)
    if missing:
        raise ValueError(f"missing arrays: {sorted(missing)}")
    n = len(arrays["actions"])
    for key, (dtype, shape) in SCHEMA.items():
        array = arrays[key]
        if array.dtype != dtype or array.shape != (n,) + shape:
            raise ValueError(f"{key}: {array.dtype}{array.shape}, "
                             f"want {np.dtype(dtype)}{(n,) + shape}")
    if arrays["actions"].max(initial=0) > PASS:
        raise ValueError("action index out of range")


def pack(episode_files: list[Path], out_dir: Path, chunk: int = CHUNK_STEPS) -> list[Path]:
    """Concatenate episode files into PokeDreamer-style fixed-size chunks.

    transitions_0001.npz, ... each `chunk` steps except the last. Episodes
    may straddle a chunk boundary, exactly as in PokeDreamer; episode_starts
    says where each begins.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    pending: dict[str, list[np.ndarray]] = {key: [] for key in SCHEMA}
    held = 0

    def flush(n: int) -> None:
        nonlocal held
        block = {key: np.concatenate(parts) for key, parts in pending.items()}
        out = {key: array[:n] for key, array in block.items()}
        check(out)
        path = out_dir / f"transitions_{len(written) + 1:04d}.npz"
        np.savez_compressed(path, **out)
        written.append(path)
        for key, array in block.items():
            pending[key] = [array[n:]]
        held -= n

    for path in episode_files:
        with np.load(path) as episode:
            for key in SCHEMA:
                pending[key].append(episode[key])
            held += len(episode["actions"])
        while held >= chunk:
            flush(chunk)
    if held:
        flush(held)
    return written
