"""A dense ladder of subgoals taken from the demonstration route itself.

The milestone chain is too coarse to plan against. `pikachu`'s goal image is
the screen at the end of a 1,485-action route; a 14-action plan towards it has
no usable gradient, and the observed failure is exactly that -- the planner
walked towards the stairs, drove predicted cost from 0.947 to 0.164, then
parked in the corner mashing A while the cost kept falling. Cost fell, progress
stopped. A goal that far away does not shape the search.

The demonstration already contains every intermediate screen. Taking its frames
every `stride` actions gives a ladder of subgoals that *are* within a horizon or
two of each other, so each plan has something close enough to steer by. Nothing new
is collected: the rungs are frames the game already produced, sitting in the
dataset the model trained on.

Advancing is deliberately not "closest rung wins". A latent distance can dip on
a screen that merely resembles a later rung -- a text box looks much like
another text box -- so a rung is cleared only when the live embedding is within
`tolerance` of it, and the ladder never moves backwards.
"""

from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np


@dataclass(frozen=True)
class Ladder:
    """Frame indices into a dataset, in route order."""

    indices: tuple[int, ...]
    episode: str
    stride: int

    def __len__(self) -> int:
        return len(self.indices)


def demo_episode(data_path: str, name: str | None = None) -> tuple[int, int, str]:
    """(start, length, name) of a successful demonstration episode."""
    with h5py.File(data_path, "r") as file:
        episodes = file["episodes"][:]
        meta = file["episode_meta"][:]

    candidates = [
        (int(start), int(length), row["name"].decode())
        for (start, length), row in zip(episodes, meta)
        if row["success"] and row["kind"] == b"demo"
    ]
    if not candidates:
        raise RuntimeError(
            f"{data_path} has no successful demo episode to build a ladder from. "
            "Collect with scripts/collect.py --demos N first."
        )
    if name is not None:
        for start, length, found in candidates:
            if found == name:
                return start, length, found
        raise ValueError(f"no demo episode named {name!r} in {data_path}")
    # The longest demonstration covers the most of the route.
    return max(candidates, key=lambda row: row[1])


def build(data_path: str, stride: int = 20, name: str | None = None) -> Ladder:
    """Rungs every `stride` frames along a demonstration, ending at its goal."""
    if stride < 1:
        raise ValueError("stride must be positive")
    start, length, episode = demo_episode(data_path, name)

    indices = list(range(start, start + length, stride))
    last = start + length - 1
    if indices[-1] != last:
        indices.append(last)  # the goal frame is always the final rung
    # The first rung is the start state itself, which is already satisfied.
    if len(indices) > 1:
        indices = indices[1:]
    return Ladder(tuple(indices), episode, stride)


def dedupe(data_path: str, ladder: Ladder) -> Ladder:
    """Drop rungs byte-identical to the previous one.

    Long dialogue stretches hold the same screen for many actions, so a fixed
    stride produces runs of identical rungs that each have to be "cleared"
    separately while the planner has nothing to do.
    """
    keep: list[int] = []
    previous: bytes | None = None
    with h5py.File(data_path, "r") as file:
        for index in ladder.indices:
            key = file["frames"][index, :, :, 0].tobytes()
            if key != previous:
                keep.append(index)
            previous = key
    return Ladder(tuple(keep), ladder.episode, ladder.stride)


def load_frames(data_path: str, ladder: Ladder) -> np.ndarray:
    with h5py.File(data_path, "r") as file:
        return np.stack([file["frames"][i, :, :, 0] for i in ladder.indices])
