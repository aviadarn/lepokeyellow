"""Pick start and goal frames out of a collected dataset.

Upstream finds goal frames by looking for episodes named after the three Red
starters and scanning `party_counts` for the first positive entry. Neither
survives the move to Yellow: there are no three starters, and `leave_room`
succeeds without the party changing at all.

Both come from the per-step `goal_flags` written by the collector, which is
the task's own predicate evaluated in the emulator.
"""

from __future__ import annotations

import h5py
import numpy as np
import torch as t


def dataset_task(data_path: str) -> tuple[str, str]:
    with h5py.File(data_path, "r") as file:
        return str(file.attrs.get("game", "")), str(file.attrs.get("task", ""))


def check_dataset(data_path: str, game: str, task: str) -> None:
    found_game, found_task = dataset_task(data_path)
    if found_game and found_game != game:
        raise ValueError(f"{data_path} was collected on {found_game!r}, not {game!r}")
    if found_task and found_task != task:
        raise ValueError(f"{data_path} was collected for task {found_task!r}, not {task!r}")


def start_index(data_path: str) -> int:
    """First frame of the first episode: the saved start state."""
    with h5py.File(data_path, "r") as file:
        return int(file["episodes"][0, 0])


def goal_indices(data_path: str, limit: int = 8, kinds=(b"demo", b"noisy")) -> list[int]:
    """First goal-satisfying frame of each successful episode.

    Prefers demonstrations and their perturbations over lucky random episodes:
    a random walk that stumbled into the goal tends to arrive in an unusual
    screen state, and a goal set is only as tight as its worst member.
    """
    with h5py.File(data_path, "r") as file:
        episodes = file["episodes"][:]
        meta = file["episode_meta"][:]
        if "goal_flags" not in file:
            raise RuntimeError(
                f"{data_path} has no goal_flags dataset; it was written by an "
                "older collector. Re-collect with scripts/collect.py."
            )
        flags = file["goal_flags"][:]

    preferred, fallback = [], []
    for (start, length), row in zip(episodes, meta):
        if not row["success"]:
            continue
        window = flags[int(start) : int(start) + int(length)]
        positive = np.flatnonzero(window)
        if not len(positive):
            continue
        index = int(start) + int(positive[0])
        (preferred if row["kind"] in kinds else fallback).append(index)

    chosen = preferred or fallback
    if not chosen:
        raise RuntimeError(
            f"{data_path} contains no successful episode, so there is no goal "
            "embedding to plan towards. Check that the route reaches the goal."
        )
    # Spread the picks across episodes rather than taking the first few.
    step = max(1, len(chosen) // limit)
    return chosen[::step][:limit]


def load_frame(data_path: str, index: int) -> t.Tensor:
    with h5py.File(data_path, "r") as file:
        frame = file["frames"][index, :, :, 0].astype(np.float32)
    frame /= 127.5
    frame -= 1.0
    return t.from_numpy(frame).unsqueeze(0).unsqueeze(0)


def default_device() -> str:
    if t.cuda.is_available():
        return "cuda"
    if getattr(t.backends, "mps", None) is not None and t.backends.mps.is_available():
        return "mps"
    return "cpu"
