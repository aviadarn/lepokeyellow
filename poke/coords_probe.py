"""Read the player's tile out of the frozen encoder, and plan on that.

Measured on 31,759 frames: within-room position is recoverable from the encoder
on every map-axis pair. So a planner does not need a latent norm -- it can be
scored on where it thinks the player will END UP, in tiles, which is a quantity
with units and a meaning.

Two heads, both on frozen embeddings:

  map       one-vs-rest ridge over the maps in the data, because the coordinate
            frames are per-map: (6, 3) in the bedroom and (6, 3) in Pallet Town
            are unrelated places
  y, x      one regressor per map, fitted only on that map's frames

The y/x head is `head="mlp"` by default, and that choice is the difference
between a planner that moves and one that does not. With ridge, held-out error
in Pallet Town was 1.88 tiles in y and 1.44 in x, while adjacent levels of the
BFS cost field differ by exactly one hop -- so the planner was choosing between
options whose true costs differed by less than the error of the instrument
measuring them. Measured directly, it could not: predicted x sat in 6.8-7.5 for
all seven buttons while the truth was 5. Two hidden layers on the same frozen
embeddings cut that to 0.53 and 0.48 (results/probe_nonlinear.json), under the
one-tile granularity the cost field needs. The map head stays linear: it was
already accurate and it is not what was failing.

Scoring a predicted state means: which map does the model think this is, and
where in it. A plan that lands on the wrong map is penalised outright rather
than compared on coordinates that would not be commensurate.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field

import h5py
import numpy as np
import torch as t

from poke.probe_linear import fit_ridge
from poke.probe_mlp import MLPHead, apply_stack, fit_mlp, stack
from poke.splits import frame_episodes


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
    head: str = "linear"
    linear_tile_mae: float = float("nan")
    y_heads: list[MLPHead] = dataclass_field(default_factory=list)
    x_heads: list[MLPHead] = dataclass_field(default_factory=list)
    frames_per_map: dict[int, int] = dataclass_field(default_factory=dict)
    # Held-out error per map and axis, e.g. {"map0_y": 0.53}. The aggregate
    # hides the only one that matters: the bedroom is 76% of the data and
    # reads to a tenth of a tile, so a mean across maps looks excellent while
    # the one room the planner has to cross is five times worse.
    holdout_tile_mae: dict[str, float] = dataclass_field(default_factory=dict)
    fitted_on: str = "real"

    def to_torch(self, device):
        as_t = lambda a: t.from_numpy(np.ascontiguousarray(a)).float().to(device)
        out = {
            "map_w": as_t(self.map_w), "map_b": as_t(self.map_b),
            "y_w": as_t(self.y_w), "y_b": as_t(self.y_b),
            "x_w": as_t(self.x_w), "x_b": as_t(self.x_b),
            "maps": t.tensor(self.maps, device=device),
        }
        if self.head == "mlp":
            out["y_mlp"] = stack(self.y_heads, device)
            out["x_mlp"] = stack(self.x_heads, device)
        return out


def _encode(model, frames, device, batch=256):
    out = []
    with t.inference_mode():
        for start in range(0, len(frames), batch):
            chunk = frames[start : start + batch].astype(np.float32) / 127.5 - 1.0
            out.append(model.encode(t.from_numpy(chunk).unsqueeze(1).to(device)).cpu().numpy())
    return np.concatenate(out)


def _constant_head(like: MLPHead, value: float) -> MLPHead:
    """A head of the same shape that always answers `value`.

    stack() needs every map's head to have identical shapes, and a map with too
    few frames or a constant coordinate has nothing to fit. Zero weights with
    the mean in the output bias is the linear path's fallback, in MLP clothing.
    """
    return MLPHead(
        feature_mean=np.zeros_like(like.feature_mean),
        feature_scale=np.ones_like(like.feature_scale),
        weights=[np.zeros_like(w) for w in like.weights],
        biases=[np.zeros_like(b) for b in like.biases[:-1]]
        + [np.full_like(like.biases[-1], value)],
        epochs_used=0,
        val_mae=float("nan"),
    )


def _sample_rows(location: np.ndarray, max_per_map: int, seed: int) -> np.ndarray:
    """Cap each map's frames separately, rather than subsampling globally.

    A global cap starves the map that matters: Pallet Town is 3,233 of 31,759
    frames because every random-walk episode starts in the bedroom, so a flat
    12,000-frame sample leaves it about 1,200 -- the one room the planner has
    to cross, fitted on the least data.
    """
    usable = np.flatnonzero(location[:, 0] >= 0)
    rng = np.random.default_rng(seed)
    kept = []
    for map_id in sorted(set(location[usable, 0].tolist())):
        rows = usable[location[usable, 0] == map_id]
        if len(rows) > max_per_map:
            rows = rng.choice(rows, max_per_map, replace=False)
        kept.append(rows)
    return np.sort(np.concatenate(kept))


def _predicted_features(model, file, rows, steps, location, owner, device,
                        batch: int = 512):
    """Features the PLANNER will see: k-step predicted embeddings.

    The read-out is fitted on real encodings and then applied to the
    predictor's output, and those are not the same distribution. Measured on
    held-out episodes (results/rollout_tiles.json): the head reads a real
    screen to 0.21 tiles and a ONE-STEP prediction of that same screen to
    2.34, with barely any further decay out to eight steps. That is not drift,
    it is an offset -- the probe is being asked to read a point it was never
    fitted near. So fit it there instead: feature = the embedding predicted
    after k recorded actions, label = where the player actually was then.

    That works, and it is not what a planner wants. Doing it cuts one-step
    error from 2.34 tiles to 0.78 and makes the decisions WORSE: the action
    with the lowest predicted cost was the best available 27.5% of the time
    with a head fitted on real encodings and 5.0% with this one, and it
    strictly reduced the true distance 47.5% of the time against 0.0%
    (results/nav_decisions_mlp_real.json against _mlp_both.json). Fitting on
    predicted features teaches the head the conditional mean of position given
    a blurred prediction, which is exactly the calibration that throws away
    the differences BETWEEN actions -- and a planner only ever needs the
    ranking. So this is kept, measured, and off by default.
    """
    episodes = file["episodes"][:]
    end_of = np.zeros(len(owner), dtype=np.int64)
    bounds = {index: int(start) + int(length)
              for index, (start, length) in enumerate(episodes)}
    for position, frame in enumerate(rows):
        end_of[position] = bounds[int(owner[position])]

    actions = file["actions"][:].astype(np.float32)
    features, labels, groups = [], [], []
    for k in range(1, steps + 1):
        # Bound the index before reading it: `rows + k` runs past the end of
        # the dataset for the last frames of the last episode, and an `&` does
        # not stop numpy evaluating both sides.
        inside = np.flatnonzero(rows + k < end_of)
        keep = inside[location[rows[inside] + k, 0] >= 0]
        if not len(keep):
            continue
        starts = rows[keep]
        frames = file["frames"][:, :, :, 0][starts]
        with t.inference_mode():
            for begin in range(0, len(starts), batch):
                block = frames[begin : begin + batch].astype(np.float32) / 127.5 - 1.0
                initial = model.encode(t.from_numpy(block).unsqueeze(1).to(device))
                taken = np.stack([
                    actions[index : index + k]
                    for index in starts[begin : begin + batch]
                ])
                predicted = model.rollout(initial, t.from_numpy(taken).to(device))
                features.append(predicted[:, -1].cpu().numpy())
        labels.append(location[starts + k])
        groups.append(owner[keep])
    return (np.concatenate(features).astype(np.float64),
            np.concatenate(labels), np.concatenate(groups))


def fit(model, data_path: str, device, alpha: float = 100.0,
        max_per_map: int = 6000, seed: int = 0, head: str = "mlp",
        min_frames: int = 64, only_episodes=None, on: str = "real",
        steps: int = 2, **mlp_kwargs) -> CoordProbe:
    """Fit the probe.

    `only_episodes` restricts it to those episode indices, which is how a
    measurement keeps episodes the probe never saw. `on` chooses what the head
    is fitted on: "real" encodings, "predicted" embeddings `steps` actions
    ahead, or "both". The planner only ever decodes predicted embeddings.
    """
    if head not in ("linear", "mlp"):
        raise ValueError(f"head must be 'linear' or 'mlp', not {head!r}")
    if on not in ("real", "predicted", "both"):
        raise ValueError(f"on must be real, predicted or both, not {on!r}")

    with h5py.File(data_path, "r") as file:
        if "location" not in file:
            raise RuntimeError(
                f"{data_path} has no location dataset; re-collect with the "
                "current scripts/collect.py (schema 3)."
            )
        location = file["location"][:]
        all_owners = frame_episodes(file["episodes"][:], file["frames"].shape[0])
        if only_episodes is not None:
            location = location.copy()
            location[~np.isin(all_owners, np.asarray(only_episodes)), 0] = -1
        rows = _sample_rows(location, max_per_map, seed)
        owner = all_owners[rows]

        frames = file["frames"][:, :, :, 0][rows]
        real = (_encode(model, frames, device).astype(np.float64),
                location[rows], owner)
        parts = [real] if on in ("real", "both") else []
        if on in ("predicted", "both"):
            parts.append(_predicted_features(model, file, rows, steps,
                                             location, owner, device))

    embeddings = np.concatenate([features for features, _, _ in parts])
    labels = np.concatenate([label for _, label, _ in parts])
    owner = np.concatenate([group for _, _, group in parts])
    location = labels
    maps = sorted(set(location[:, 0].tolist()))

    # The map head is fitted on real encodings whatever `on` says, because the
    # two heads fail differently and only one of them was failing. Adding
    # predicted features to the map head cost the one call that matters: from
    # outside the player's front door, `up` walks into the house, and the
    # map head trained on real encodings predicts that correctly while the
    # mixed one calls it Pallet Town and reads the tile as the north edge --
    # a cost of 1 for the action that ends the episode. Greedy argmin is
    # decided by the best-looking option, so one confident error outweighs a
    # better average.
    map_w, map_b = [], []
    for map_id in maps:
        w, b = fit_ridge(real[0], (real[1][:, 0] == map_id).astype(np.float64),
                         alpha)
        map_w.append(w)
        map_b.append(b)
    map_w = np.stack(map_w, axis=1)
    map_b = np.array(map_b)

    scores = real[0] @ map_w + map_b
    predicted_map = np.array(maps)[scores.argmax(axis=1)]
    accuracy = float((predicted_map == real[1][:, 0]).mean())

    y_w, y_b, x_w, x_b, errors = [], [], [], [], []
    y_heads: list[MLPHead] = []
    x_heads: list[MLPHead] = []
    holdout: list[float] = []
    per_map: dict[str, float] = {}
    for map_id in maps:
        on_map = location[:, 0] == map_id
        for axis, store_w, store_b, heads in (
            (1, y_w, y_b, y_heads),
            (2, x_w, x_b, x_heads),
        ):
            target = location[on_map, axis].astype(np.float64)
            fittable = on_map.sum() >= min_frames and target.std() >= 1e-6
            if not fittable:
                store_w.append(np.zeros(embeddings.shape[1]))
                store_b.append(float(target.mean()) if on_map.sum() else 0.0)
            else:
                w, b = fit_ridge(embeddings[on_map], target, alpha)
                store_w.append(w)
                store_b.append(b)
                errors.append(float(np.abs(embeddings[on_map] @ w + b - target).mean()))
            if head == "mlp":
                if fittable:
                    fitted = fit_mlp(
                        embeddings[on_map], target, groups=owner[on_map],
                        seed=seed, device=device, **mlp_kwargs,
                    )
                    heads.append(fitted)
                    holdout.append(fitted.val_mae)
                    per_map[f"map{map_id}_{'y' if axis == 1 else 'x'}"] = (
                        fitted.val_mae
                    )
                else:
                    # Shape has to match the other heads; fill it in after the
                    # loop, once a real head exists to copy the shape from.
                    heads.append(float(target.mean()) if on_map.sum() else 0.0)

    linear_mae = float(np.mean(errors)) if errors else float("nan")
    if head == "mlp":
        template = next(
            (h for h in y_heads + x_heads if isinstance(h, MLPHead)), None
        )
        if template is None:
            raise RuntimeError(
                f"{data_path} has no map with at least {min_frames} frames and "
                "a varying coordinate, so there is nothing to fit."
            )
        y_heads = [h if isinstance(h, MLPHead) else _constant_head(template, h)
                   for h in y_heads]
        x_heads = [h if isinstance(h, MLPHead) else _constant_head(template, h)
                   for h in x_heads]

    return CoordProbe(
        maps=maps,
        map_w=map_w, map_b=map_b,
        y_w=np.stack(y_w), y_b=np.array(y_b),
        x_w=np.stack(x_w), x_b=np.array(x_b),
        map_accuracy=accuracy,
        # The linear number is error on the frames it was fitted on; the MLP's
        # is error on episodes it never saw, because early stopping needs a
        # held-out split anyway. They are not the same kind of number, and the
        # harder one is the one that is better.
        tile_mae=float(np.mean(holdout)) if head == "mlp" else linear_mae,
        predicted_tile_mae=float("nan"),
        head=head,
        linear_tile_mae=linear_mae,
        y_heads=y_heads if head == "mlp" else [],
        x_heads=x_heads if head == "mlp" else [],
        frames_per_map={int(m): int((location[:, 0] == m).sum()) for m in maps},
        holdout_tile_mae=per_map,
        fitted_on=on,
    )


def decode(states: t.Tensor, probe: dict, chunk: int = 8192,
           map_index: t.Tensor | None = None) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    """[..., D] embeddings -> (map index, y, x), all same leading shape.

    Chunked because the MLP head evaluates every map's head on every row before
    it knows which map each row is on, so the intermediate is n_maps x hidden
    wide and a full CEM rollout is tens of thousands of rows.
    """
    flat = states.reshape(-1, states.shape[-1])
    # A measurement may want the coordinate head of a map it already knows,
    # so that positional error is not mixed with map-classification error.
    given = None if map_index is None else map_index.reshape(-1)
    maps_out, ys, xs = [], [], []
    for start in range(0, flat.shape[0], chunk):
        block = flat[start : start + chunk]
        map_index = (
            given[start : start + chunk] if given is not None
            else (block @ probe["map_w"] + probe["map_b"]).argmax(dim=-1)
        )
        if "y_mlp" in probe:
            per_map_y = apply_stack(block, probe["y_mlp"])
            per_map_x = apply_stack(block, probe["x_mlp"])
        else:
            per_map_y = block @ probe["y_w"].T + probe["y_b"]
            per_map_x = block @ probe["x_w"].T + probe["x_b"]
        maps_out.append(map_index)
        ys.append(per_map_y.gather(1, map_index[:, None]).squeeze(1))
        xs.append(per_map_x.gather(1, map_index[:, None]).squeeze(1))

    shape = states.shape[:-1]
    return (t.cat(maps_out).reshape(shape),
            t.cat(ys).reshape(shape),
            t.cat(xs).reshape(shape))


def aggregate_cost(distance: t.Tensor, how: str = "discounted",
                   discount: float = 0.9) -> t.Tensor:
    """Collapse a [samples, horizon] cost trace into one cost per plan.

    `min` takes the best state anywhere in the rollout, inherited from the goal
    objective. It is the wrong rule for a learned read-out: over a 28-action
    lookahead, the cost of a plan becomes the single most optimistic tile the
    head decoded anywhere along it, and the head's error grows with the rollout
    length (0.21 tiles on a real screen, 2.34 at one step, 3.06 at eight).
    Searching 384 sequences for the lowest such number is a search for the
    head's largest mistake.

    `final` scores only where the plan ends, and `discounted` weights the trace
    by `discount ** t`, which keeps the steps about to be committed decisive
    while still preferring a plan that keeps improving.
    """
    if how == "min":
        return distance.min(dim=1).values
    if how == "final":
        return distance[:, -1]
    if how != "discounted":
        raise ValueError(f"unknown cost aggregation {how!r}")
    weights = discount ** t.arange(
        distance.size(1), device=distance.device, dtype=distance.dtype
    )
    return (distance * weights).sum(dim=1) / weights.sum()


def plan_costs(model, initial, action_sequences, probe, lookup,
               target_map_index: int, target_y: float, target_x: float,
               wrong_map_penalty: float = 50.0, repeat: int = 1,
               field: t.Tensor | None = None, how: str = "discounted",
               discount: float = 0.9):
    """Cost of each plan, in BFS hops over observed transitions or in tiles.

    `how` decides how a plan's cost trace becomes one number; see
    `aggregate_cost`, where the choice matters more than it looks.
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
    elif field.ndim == 3:
        # One cost grid per map, searched across doorways, so the wrong room
        # is not a flat penalty but a real distance -- and a plan that walks
        # into a building can be scored on how to get back out. Without it,
        # every action inside the player's house looked identical and the
        # planner sat there (results/nav_decisions_mlp_both.json).
        rows, cols = field.shape[1:]
        yi = y.round().long().clamp(0, rows - 1)
        xi = x.round().long().clamp(0, cols - 1)
        return aggregate_cost(field[map_index, yi, xi], how, discount)
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
    return aggregate_cost(distance, how, discount)
