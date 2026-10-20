"""Windowed frame/action dataset. Adapted from stmonty/lePokeRed.

Changed: an episode set with no window long enough raises a message naming the
longest episode, instead of numpy's "need at least one array to concatenate".
`finetune_rollout --max-horizon` is the usual way to hit this.
"""

from pathlib import Path

import numpy as np
import torch
from h5py import File
from torch.utils.data import Dataset


class PokeDataset(Dataset):
    def __init__(
        self,
        data_path: str | Path,
        num_frames: int = 4,
        episode_indices=None,
    ) -> None:
        self.data_path = Path(data_path)
        self.num_frames = num_frames
        self._h5 = None

        with File(self.data_path, "r") as f:
            episodes = f["episodes"][:]

        if episode_indices is not None:
            episodes = episodes[np.asarray(episode_indices, dtype=np.int64)]

        # a window starting at s needs frames s .. s+num_frames-1 inside the episode
        starts = []
        for start, length in episodes:
            n = int(length) - num_frames + 1
            if n > 0:
                starts.append(np.arange(int(start), int(start) + n, dtype=np.int64))
        if not starts:
            longest = int(episodes[:, 1].max()) if len(episodes) else 0
            raise ValueError(
                f"{self.data_path}: no episode is long enough for a "
                f"{num_frames}-frame window (longest episode is {longest} "
                "frames). Collect longer episodes, or lower num_frames / "
                "--max-horizon."
            )
        self.starts = np.concatenate(starts)

    def _file(self) -> File:
        if self._h5 is None:
            self._h5 = File(self.data_path, "r")
        return self._h5

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        f = self._file()
        s = int(self.starts[idx])
        e = s + self.num_frames

        frames = f["frames"][s:e, :, :, 0].astype(np.float32) / 127.5 - 1.0
        actions = f["actions"][s:e].astype(np.float32)

        batch = {
            "frames": torch.from_numpy(frames).unsqueeze(1),
            "actions": torch.from_numpy(actions),
        }
        # Schema 3 records (map, y, x) per frame, with -1 where the read
        # failed. The auxiliary coordinate head needs it; older datasets
        # simply do not carry it and train without that loss term.
        if "location" in f:
            batch["location"] = torch.from_numpy(
                f["location"][s:e].astype(np.int64)
            )
        return batch

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None
