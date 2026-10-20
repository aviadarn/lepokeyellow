"""A non-linear read-out head for the frozen encoder.

Why this exists. The linear coordinate probe reads the player's tile out of the
encoder to within 0.22-0.78 tiles in the three indoor maps, and within
1.44 (x) and 1.88 (y) tiles in Pallet Town. Adjacent levels of the BFS cost
field differ by exactly one hop, so in Pallet the planner is being asked to
choose between options whose true costs differ by less than the error of the
instrument measuring them -- and measured directly, it cannot: the model
predicts x in 6.8-7.5 for all seven buttons while the truth is 5.

That is not evidence the encoder lacks the information. Under the SAME linear
probe, raw 8x-downsampled pixels score R^2 0.912/0.803 on Pallet against the
encoder's 0.760/0.655. A linear map of 144x160 greyscale beats a linear map of
the encoder's 192 dimensions, so the encoder is not the bottleneck; the linear
read-out is. Raw pixels cannot replace it for planning -- the world model
predicts future EMBEDDINGS, not future screens, so whatever scores a plan has
to be a function of an embedding -- but they bound what a better head could
recover.

So: same frozen embeddings, two hidden layers instead of none.

ReLU rather than GELU, and the target standardisation folded into the output
layer after fitting, so one set of weights evaluates identically through numpy
(for measurement, in scripts/probe_nonlinear.py) and through torch (for
planning, batched on-device in poke.coords_probe).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch as t


@dataclass
class MLPHead:
    """Standardise, then `len(weights)` affine layers with ReLU between."""

    feature_mean: np.ndarray      # [D]
    feature_scale: np.ndarray     # [D]
    weights: list[np.ndarray]     # [D, H], [H, H], [H, 1]
    biases: list[np.ndarray]
    epochs_used: int
    val_mae: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        hidden = (np.asarray(x, dtype=np.float64) - self.feature_mean) / self.feature_scale
        last = len(self.weights) - 1
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            hidden = hidden @ weight + bias
            if index < last:
                hidden = np.maximum(hidden, 0.0)
        return hidden[:, 0]


def _blocks(n: int, size: int = 100) -> np.ndarray:
    """Fallback grouping when the caller has no episode ids.

    Frames 60 emulator frames apart are nearly identical, so a random frame
    split would put a frame's own neighbours on both sides of the inner
    validation split and early stopping would run far too long.
    """
    return np.arange(n) // size


def _inner_split(groups: np.ndarray, val_fraction: float, seed: int):
    """Hold out whole groups, so early stopping is not read off leaked frames."""
    unique = np.unique(groups)
    if len(unique) < 2:
        cut = int(len(groups) * (1.0 - val_fraction))
        mask = np.zeros(len(groups), dtype=bool)
        mask[max(cut, 1) :] = True
        return ~mask, mask
    shuffled = unique.copy()
    np.random.default_rng(seed).shuffle(shuffled)
    held = shuffled[: max(1, int(round(len(unique) * val_fraction)))]
    validation = np.isin(groups, held)
    if validation.all():
        validation = np.isin(groups, held[:-1]) if len(held) > 1 else validation
    return ~validation, validation


def fit_mlp(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None = None,
    hidden: int = 256,
    depth: int = 2,
    epochs: int = 400,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-2,
    batch_size: int = 256,
    val_fraction: float = 0.2,
    patience: int = 60,
    seed: int = 0,
    device=None,
) -> MLPHead:
    """Fit a small MLP on frozen features, early-stopped on held-out groups.

    `groups` are episode ids. The split for early stopping holds out whole
    episodes; without that, the stopping epoch is chosen on frames whose
    neighbours are in the training set and the head is left overfitted.
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if depth < 1:
        raise ValueError("depth must be at least 1")

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    groups = _blocks(len(y)) if groups is None else np.asarray(groups)

    feature_mean = x.mean(axis=0)
    feature_scale = x.std(axis=0)
    feature_scale[feature_scale < 1e-8] = 1.0
    target_mean = float(y.mean())
    target_scale = float(y.std()) or 1.0

    standardised = (x - feature_mean) / feature_scale
    targets = (y - target_mean) / target_scale

    train_rows, val_rows = _inner_split(groups, val_fraction, seed)
    device = t.device("cpu") if device is None else device
    t.manual_seed(seed)

    sizes = [x.shape[1]] + [hidden] * depth + [1]
    modules: list[t.nn.Module] = []
    for index in range(len(sizes) - 1):
        modules.append(t.nn.Linear(sizes[index], sizes[index + 1]))
        if index < len(sizes) - 2:
            modules.append(t.nn.ReLU())
    model = t.nn.Sequential(*modules).to(device)

    as_tensor = lambda a: t.from_numpy(np.ascontiguousarray(a)).float().to(device)
    xt = as_tensor(standardised[train_rows])
    yt = as_tensor(targets[train_rows])[:, None]
    xv = as_tensor(standardised[val_rows])
    yv = as_tensor(targets[val_rows])[:, None]

    optimiser = t.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = t.Generator(device="cpu").manual_seed(seed)

    best_mae = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        order = t.randperm(xt.size(0), generator=generator).to(device)
        for start in range(0, xt.size(0), batch_size):
            rows = order[start : start + batch_size]
            optimiser.zero_grad(set_to_none=True)
            loss = t.nn.functional.mse_loss(model(xt[rows]), yt[rows])
            loss.backward()
            optimiser.step()

        model.eval()
        with t.no_grad():
            mae = (model(xv) - yv).abs().mean().item()
        if mae < best_mae - 1e-6:
            best_mae = mae
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
        elif epoch - best_epoch >= patience:
            break

    model.load_state_dict(best_state)
    weights, biases = [], []
    for layer in model:
        if isinstance(layer, t.nn.Linear):
            weights.append(layer.weight.detach().cpu().numpy().astype(np.float64).T)
            biases.append(layer.bias.detach().cpu().numpy().astype(np.float64))

    # Fold the target standardisation into the output layer, so the head reads
    # out tiles directly and nothing downstream has to carry y's mean and std.
    weights[-1] = weights[-1] * target_scale
    biases[-1] = biases[-1] * target_scale + target_mean

    return MLPHead(
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        weights=weights,
        biases=biases,
        epochs_used=best_epoch,
        val_mae=best_mae * target_scale,
    )


def stack(heads: list[MLPHead], device) -> dict:
    """Batch several same-shaped heads into one set of tensors.

    The planner decodes a rollout of [samples, horizon, D] embeddings and needs
    every map's head evaluated before it knows which map each state is on, so
    the heads run as one batched einsum rather than a Python loop over maps.
    """
    if not heads:
        raise ValueError("stack needs at least one head")
    shapes = {tuple(w.shape for w in head.weights) for head in heads}
    if len(shapes) != 1:
        raise ValueError(f"heads have mismatched shapes: {shapes}")

    as_tensor = lambda a: t.from_numpy(np.ascontiguousarray(a)).float().to(device)
    return {
        "mean": as_tensor(np.stack([h.feature_mean for h in heads])),
        "scale": as_tensor(np.stack([h.feature_scale for h in heads])),
        "weights": [
            as_tensor(np.stack([h.weights[layer] for h in heads]))
            for layer in range(len(heads[0].weights))
        ],
        "biases": [
            as_tensor(np.stack([h.biases[layer] for h in heads]))
            for layer in range(len(heads[0].biases))
        ],
    }


def apply_stack(features: t.Tensor, stacked: dict) -> t.Tensor:
    """[N, D] features -> [N, n_heads], every head evaluated on every row."""
    hidden = (features[:, None, :] - stacked["mean"][None]) / stacked["scale"][None]
    layers = len(stacked["weights"])
    for index, (weight, bias) in enumerate(zip(stacked["weights"], stacked["biases"])):
        hidden = t.einsum("nmd,mdh->nmh", hidden, weight) + bias[None]
        if index < layers - 1:
            hidden = t.relu(hidden)
    return hidden[:, :, 0]
