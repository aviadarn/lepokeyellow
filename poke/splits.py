from pathlib import Path

import numpy as np
from h5py import File


def episode_split(
    data_path: str | Path,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    with File(data_path, "r") as f:
        success = f["episode_meta"]["success"][:].astype(bool)

    rng = np.random.default_rng(seed)
    train, val = [], []

    # split failures and successes separately so both land in each half
    for group in (np.flatnonzero(~success), np.flatnonzero(success)):
        rng.shuffle(group)
        n = round(len(group) * val_fraction)
        val.append(group[:n])
        train.append(group[n:])

    return np.sort(np.concatenate(train)), np.sort(np.concatenate(val))


def frame_episodes(episodes: np.ndarray, total: int) -> np.ndarray:
    """Which episode each frame belongs to, as an index into `episodes`.

    A probe must be split by episode and not by frame: one episode is a
    continuous walk, so a random frame split puts a step and its neighbour on
    opposite sides and measures autocorrelation instead of generalisation.
    """
    owner = np.zeros(total, dtype=np.int64)
    for index, (start, length) in enumerate(episodes):
        owner[int(start) : int(start) + int(length)] = index
    return owner
