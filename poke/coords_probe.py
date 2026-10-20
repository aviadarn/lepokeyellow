"""Read the player's tile out of the frozen encoder, and plan on that.

Measured on 31,759 frames: within-room position is linearly recoverable from
the encoder on every map-axis pair, to within half a tile in three of four
rooms. So a planner does not need a latent norm -- it can be scored on where it
thinks the player will END UP, in tiles, which is a quantity with units and a
meaning.

Two heads, both ridge on frozen embeddings:

  map       one-vs-rest over the maps in the data, because the coordinate
            frames are per-map: (6, 3) in the bedroom and (6, 3) in Pallet
            Town are unrelated places
  y, x      one regressor per map, fitted only on that map's frames

Scoring a predicted state means: which map does the model think this is, and
where in it. A plan that lands on the wrong map is penalised outright rather
than compared on coordinates that would not be commensurate.
"""

from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np
import torch as t

from poke.probe_linear import fit_ridge


@dataclass
class CoordProbe:
    maps: list[int]
    map_w: np.ndarray        # [D, n_maps]
    map_b: np.ndarray        # [n_maps]
    y_w: np.ndarray          # [n_maps, D]
    y_b: np.ndarray
    x_w: np.ndarray
    x_b: np.ndarray
    map_accuracy: float
    tile_mae: float
    predicted_tile_mae: float

    def to_torch(self, device):
        as_t = lambda a: t.from_numpy(np.ascontiguousarray(a)).float().to(device)
        return {
            "map_w": as_t(self.map_w), "map_b": as_t(self.map_b),
            "y_w": as_t(self.y_w), "y_b": as_t(self.y_b),
            "x_w": as_t(self.x_w), "x_b": as_t(self.x_b),
            "maps": t.tensor(self.maps, device=device),
        }


def _encode(model, frames, device, batch=256):
    out = []
    with t.inference_mode():
        for start in range(0, len(frames), batch):
            chunk = frames[start : start + batch].astype(np.float32) / 127.5 - 1.0
            out.append(model.encode(t.from_numpy(chunk).unsqueeze(1).to(device)).cpu().numpy())
    return np.concatenate(out)


def fit(model, data_path: str, device, alpha: float = 100.0,
        max_frames: int = 12000, seed: int = 0) -> CoordProbe:
    with h5py.File(data_path, "r") as file:
        if "location" not in file:
            raise RuntimeError(
                f"{data_path} has no location dataset; re-collect with the "
                "current scripts/collect.py (schema 3)."
            )
        location = file["location"][:]
        usable = np.flatnonzero(location[:, 0] >= 0)
        rng = np.random.default_rng(seed)
        if len(usable) > max_frames:
            usable = np.sort(rng.choice(usable, max_frames, replace=False))
        frames = file["frames"][:, :, :, 0][usable]
    location = location[usable]

    embeddings = _encode(model, frames, device).astype(np.float64)
    maps = sorted(set(location[:, 0].tolist()))

    map_w, map_b = [], []
    for map_id in maps:
        w, b = fit_ridge(embeddings, (location[:, 0] == map_id).astype(np.float64), alpha)
        map_w.append(w)
        map_b.append(b)
    map_w = np.stack(map_w, axis=1)
    map_b = np.array(map_b)

    scores = embeddings @ map_w + map_b
    predicted_map = np.array(maps)[scores.argmax(axis=1)]
    accuracy = float((predicted_map == location[:, 0]).mean())

    y_w, y_b, x_w, x_b, errors = [], [], [], [], []
    for map_id in maps:
        rows = location[:, 0] == map_id
        for axis, store_w, store_b in ((1, y_w, y_b), (2, x_w, x_b)):
            target = location[rows, axis].astype(np.float64)
            if rows.sum() < 32 or target.std() < 1e-6:
                store_w.append(np.zeros(embeddings.shape[1]))
                store_b.append(float(target.mean()) if rows.sum() else 0.0)
                continue
            w, b = fit_ridge(embeddings[rows], target, alpha)
            store_w.append(w)
            store_b.append(b)
            errors.append(float(np.abs(embeddings[rows] @ w + b - target).mean()))

    return CoordProbe(
        maps=maps,
        map_w=map_w, map_b=map_b,
        y_w=np.stack(y_w), y_b=np.array(y_b),
        x_w=np.stack(x_w), x_b=np.array(x_b),
        map_accuracy=accuracy,
        tile_mae=float(np.mean(errors)) if errors else float("nan"),
        predicted_tile_mae=float("nan"),
    )


def decode(states: t.Tensor, probe: dict) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    """[..., D] embeddings -> (map index, y, x), all same leading shape."""
    flat = states.reshape(-1, states.shape[-1])
    map_index = (flat @ probe["map_w"] + probe["map_b"]).argmax(dim=-1)
    y = (flat @ probe["y_w"].T + probe["y_b"]).gather(1, map_index[:, None]).squeeze(1)
    x = (flat @ probe["x_w"].T + probe["x_b"]).gather(1, map_index[:, None]).squeeze(1)
    shape = states.shape[:-1]
    return map_index.reshape(shape), y.reshape(shape), x.reshape(shape)


def plan_costs(model, initial, action_sequences, probe, lookup,
               target_map_index: int, target_y: float, target_x: float,
               wrong_map_penalty: float = 50.0, repeat: int = 1,
               field: t.Tensor | None = None):
    """Cost at the closest point of the rollout: BFS hops, or tiles.

    Mirrors the goal objective's "best state anywhere in the rollout" rule, but
    in tiles rather than in a latent norm, so the number has units and a plan
    that lands on the wrong map is penalised outright instead of being compared
    on coordinates that are not commensurate between rooms.
    """
    # Each sampled slot becomes `repeat` emulator actions. CEM samples each
    # slot independently, so a 14-slot plan draws 10 consecutive "up" presses
    # with probability about 7^-10 -- it cannot even PROPOSE walking across an
    # open map, which is exactly where the planner stalls. Repeats make a run
    # of identical actions the unit of search instead of a coincidence, and
    # multiply the lookahead without multiplying the number of decisions.
    if repeat > 1:
        action_sequences = action_sequences.repeat_interleave(repeat, dim=1)
    actions = lookup[action_sequences]
    initial_states = initial.expand(action_sequences.size(0), -1)
    with t.inference_mode():
        states = model.rollout(initial_states, actions)
    map_index, y, x = decode(states, probe)
    if field is None:
        distance = (y - target_y).abs() + (x - target_x).abs()
    else:
        # Walls. Manhattan distance assumes an empty room, and Pallet Town is
        # not one: the tile due north of the start is the doorway back into the
        # player's house, so a greedy "reduce |dy| + |dx|" walks inside and
        # stays. `field` is BFS hop count to the target over transitions
        # actually observed in the data, so it routes around what is in the way.
        rows, cols = field.shape
        yi = y.round().long().clamp(0, rows - 1)
        xi = x.round().long().clamp(0, cols - 1)
        distance = field[yi, xi]
    distance = distance + wrong_map_penalty * (map_index != target_map_index).float()
    return distance.min(dim=1).values
