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


def goal_indices(
    data_path: str,
    limit: int = 8,
    kinds=(b"demo", b"noisy"),
    dedupe: bool = True,
    window: int = 0,
) -> list[int]:
    """First goal-satisfying frame of each successful episode, deduplicated.

    Prefers demonstrations and their perturbations over lucky random episodes:
    a random walk that stumbled into the goal tends to arrive in an unusual
    screen state, and a goal set is only as tight as its worst member.

    Deduplication is not an optimisation. Replaying one demonstration route N
    times from one save state gives N byte-identical goal frames, because the
    emulator is deterministic -- so a goal set that reads as "4 goals" is one
    screen counted four times, and the planner is aiming at a single point
    while appearing to have choices. Measured on the first real dataset: 4
    goal frames, 1 distinct screen.

    More demonstrations do not help, and it took a second dataset to see why.
    yellow_pikachu's goal fires when party_count flips during Oak's SCRIPTED
    dialogue, so the screen at that instant is the same however the player got
    there: five successful episodes across two collection runs still gave one
    distinct screen. A goal defined by a RAM event has exactly one goal frame
    by construction.

    `window` is the way out. The goal flag stays set for the rest of an
    episode, so frames shortly after the trigger also satisfy the goal while
    showing different screens -- the remainder of the dialogue. Those are
    genuine alternatives to steer towards rather than copies of one frame.
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
        span = flags[int(start) : int(start) + int(length)]
        positive = np.flatnonzero(span)
        if not len(positive):
            continue
        first = int(positive[0])
        # Frames from the trigger onwards all satisfy the goal, so `window`
        # extra ones give real alternatives instead of copies of one screen.
        taken = [
            int(start) + offset
            for offset in positive[: 1 + max(window, 0)]
            if offset >= first
        ]
        (preferred if row["kind"] in kinds else fallback).extend(taken)

    chosen = preferred or fallback
    if not chosen:
        raise RuntimeError(
            f"{data_path} contains no successful episode, so there is no goal "
            "embedding to plan towards. Check that the route reaches the goal."
        )

    if dedupe:
        chosen = _distinct_frames(data_path, chosen)

    # Spread the picks across episodes rather than taking the first few.
    step = max(1, len(chosen) // limit)
    return chosen[::step][:limit]


def _distinct_frames(data_path: str, indices: list[int]) -> list[int]:
    """Drop indices whose frame is byte-identical to an earlier one."""
    seen: set[bytes] = set()
    keep: list[int] = []
    with h5py.File(data_path, "r") as file:
        for index in indices:
            key = file["frames"][index, :, :, 0].tobytes()
            if key not in seen:
                seen.add(key)
                keep.append(index)
    return keep


def describe_goals(data_path: str, indices: list[int]) -> str:
    """One line on how much choice the goal set actually gives the planner."""
    distinct = len(_distinct_frames(data_path, indices))
    note = ""
    if distinct == 1:
        note = ("  (a single screen: the planner is aiming at one exact frame. "
                "More demonstrations will not help when the goal is a scripted "
                "RAM event -- the trigger frame is the same however you arrive. "
                "Use goal_indices(window=N) to take frames after the trigger.)")
    return f"{len(indices)} goal frame(s), {distinct} distinct{note}"


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
