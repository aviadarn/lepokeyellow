"""Ridge regression and blocked splits, for probing a frozen encoder.

Closed-form ridge in numpy, so the project keeps its dependency list. The
blocking matters more than the estimator: frames 60 emulator frames apart are
nearly identical, so a random train/test split puts a frame's own neighbours in
both halves and reports a score that measures autocorrelation rather than
generalisation. Contiguous blocks with a purge gap on each side are the
standard fix for an autocorrelated sequence.
"""

from __future__ import annotations

import numpy as np


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> tuple[np.ndarray, float]:
    """Closed-form ridge on centred features. Returns (weights, intercept)."""
    x_mean, y_mean = x.mean(axis=0), y.mean()
    xc, yc = x - x_mean, y - y_mean
    gram = xc.T @ xc + alpha * np.eye(xc.shape[1])
    weights = np.linalg.solve(gram, xc.T @ yc)
    return weights, float(y_mean - x_mean @ weights)


def predict(x: np.ndarray, weights: np.ndarray, intercept: float) -> np.ndarray:
    return x @ weights + intercept


def blocked_folds(
    n: int,
    folds: int = 5,
    purge: int = 10,
    blocks: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Blocked CV with purging, optionally interleaved.

    Purging is what stops a test block's immediate neighbours sitting in the
    training set. Without it a probe on 60-frame-apart screenshots scores well
    by interpolating between two almost identical training frames.

    `blocks` splits the sequence into that many contiguous chunks and deals
    them round-robin to the folds, instead of giving each fold one long
    contiguous span. That matters whenever the target varies along the
    sequence: with `folds` long spans and a target like "index along the
    route", fold 0 tests positions 0-93 having trained only on 104-466, so the
    probe has to EXTRAPOLATE and ridge is terrible at it. Measured on the real
    encoder that produced R^2 = -25 while the same features beat the mean on
    absolute error -- a design artefact, not a finding. Interleaving keeps
    every fold's training set spanning the whole route, so the probe is asked
    to interpolate between distant segments while still never seeing a test
    frame's neighbours.
    """
    if folds < 2:
        raise ValueError("folds must be at least 2")
    blocks = folds if blocks is None else blocks
    if blocks < folds:
        raise ValueError("blocks must be at least folds")

    bounds = np.linspace(0, n, blocks + 1).astype(int)
    spans = [(int(bounds[i]), int(bounds[i + 1])) for i in range(blocks)]

    out = []
    for fold in range(folds):
        mine = spans[fold::folds]
        test = np.concatenate([np.arange(a, b) for a, b in mine])
        keep = np.ones(n, dtype=bool)
        for a, b in mine:
            keep[max(0, a - purge) : min(n, b + purge)] = False
        out.append((np.flatnonzero(keep), test))
    return out


def r2(truth: np.ndarray, guess: np.ndarray) -> float:
    """R^2 against the mean of the TEST block.

    Scoring against the training mean would flatter a probe that only has to
    know which part of the route it is in.
    """
    residual = float(((truth - guess) ** 2).sum())
    total = float(((truth - truth.mean()) ** 2).sum())
    return 1.0 - residual / total if total > 0 else float("nan")


def cross_val(x: np.ndarray, y: np.ndarray, alpha: float = 1.0,
              folds: int = 5, purge: int = 10, blocks: int | None = None) -> dict:
    """Blocked CV. Returns R^2 and mean absolute error over held-out blocks."""
    splits = blocked_folds(len(y), folds, purge, blocks)
    predictions = np.full(len(y), np.nan)
    for train, test in splits:
        weights, intercept = fit_ridge(x[train], y[train], alpha)
        predictions[test] = predict(x[test], weights, intercept)
    scores = [r2(y[test], predictions[test]) for _, test in splits]
    return {
        "r2_per_fold": [round(float(s), 4) for s in scores],
        "r2": float(np.nanmean(scores)),
        "mae": float(np.abs(predictions - y).mean()),
        "predictions": predictions,
    }
