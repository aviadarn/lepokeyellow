"""Read route progress out of the frozen encoder, and plan on that instead.

The probe result: route position is linearly recoverable from the encoder
(R^2 0.697, +/-55 actions on a 467-action route) while downsampled raw pixels
manage 0.105 and shuffled labels sit at chance. So the embedding carries where
you are; Euclidean distance to a goal embedding was simply the wrong way to
read it.

This fits that probe and exposes it as a planning objective: a plan is good if
the states it passes through are FURTHER ALONG THE ROUTE, not if they are
numerically near a goal vector.

The catch, and it is measured rather than hoped: the probe is fitted on
embeddings the ENCODER produced from real screens, but CEM scores embeddings
the PREDICTOR produced by rolling forward. If the probe does not survive that
gap it fails silently -- every plan scores plausibly and none of it means
anything. `fit` therefore reports the probe's error on one-step predicted
embeddings alongside its error on encoded ones.
"""

from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np
import torch as t

from poke.ladder import demo_episode
from poke.probe_linear import cross_val, fit_ridge, r2


@dataclass
class ProgressProbe:
    weights: np.ndarray
    intercept: float
    length: int
    cv_r2: float
    cv_mae: float
    predicted_mae: float
    encoded_mae: float

    def torch_weights(self, device):
        return (
            t.from_numpy(self.weights).float().to(device),
            float(self.intercept),
        )

    @property
    def survives_rollout(self) -> bool:
        """Does the probe still work on predictor output, not just encoder output?

        Allowed to be worse -- a predicted embedding is an approximation -- but
        if it is no better than guessing the middle of the route it cannot
        score a plan.
        """
        return self.predicted_mae < self.length / 4


def _encode(model, frames: np.ndarray, device, batch: int = 128) -> np.ndarray:
    out = []
    with t.inference_mode():
        for start in range(0, len(frames), batch):
            chunk = frames[start : start + batch].astype(np.float32) / 127.5 - 1.0
            out.append(model.encode(t.from_numpy(chunk).unsqueeze(1).to(device)).cpu().numpy())
    return np.concatenate(out)


def fit(model, data_path: str, device, alpha: float = 100.0,
        folds: int = 5, purge: int = 10, blocks: int = 40) -> ProgressProbe:
    """Fit the position probe on a demonstration episode."""
    start, length, _ = demo_episode(data_path)
    with h5py.File(data_path, "r") as file:
        frames = file["frames"][start : start + length, :, :, 0]
        actions = file["actions"][start : start + length].astype(np.float32)

    embeddings = _encode(model, frames, device).astype(np.float64)
    position = np.arange(length, dtype=np.float64)

    scored = cross_val(embeddings, position, alpha=alpha, folds=folds,
                       purge=purge, blocks=blocks)
    weights, intercept = fit_ridge(embeddings, position, alpha)

    # The gap that matters: score embeddings the PREDICTOR produced.
    with t.inference_mode():
        current = t.from_numpy(embeddings[:-1]).float().to(device)
        action_embeddings = model.action_encoder(
            t.from_numpy(actions[:-1]).to(device).unsqueeze(1)
        )
        predicted = model.predictor(current.unsqueeze(1), action_embeddings)[:, -1]
    predicted_np = predicted.cpu().numpy().astype(np.float64)
    predicted_position = predicted_np @ weights + intercept
    encoded_position = embeddings[1:] @ weights + intercept
    truth = position[1:]

    return ProgressProbe(
        weights=weights,
        intercept=intercept,
        length=length,
        cv_r2=scored["r2"],
        cv_mae=scored["mae"],
        predicted_mae=float(np.abs(predicted_position - truth).mean()),
        encoded_mae=float(np.abs(encoded_position - truth).mean()),
    )


def plan_costs(model, initial: t.Tensor, action_sequences: t.Tensor,
               weights: t.Tensor, intercept: float, lookup: t.Tensor) -> t.Tensor:
    """CEM cost for the progress objective: minus the furthest point reached.

    Mirrors the goal objective's "best state anywhere in the rollout" rule --
    that one takes the minimum distance over the horizon, this one takes the
    maximum progress -- so the two are comparable on everything but what they
    consider good.
    """
    actions = lookup[action_sequences]
    initial_states = initial.expand(action_sequences.size(0), -1)
    with t.inference_mode():
        states = model.rollout(initial_states, actions)
    progress = states @ weights + intercept
    return -progress.max(dim=1).values
