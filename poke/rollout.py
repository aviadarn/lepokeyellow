"""Roll the predictor forward from more than one real frame of context.

`JEPA.predict_rollout` seeds the rollout with a single embedding, so the
predictor's first step sees a context of length 1. Training never does that:
`JEPA.forward` hands the predictor `num_frames - 1` consecutive embeddings and
the actions taken at each, so with the default num_frames=4 every training step
had three frames of context and planning has one. Whether that mismatch costs
anything is a measurement, not an argument -- scripts/rollout_tiles.py makes it
by decoding both rollouts into tiles and comparing against the emulator.

Alignment, which is the only thing here that is easy to get wrong: training
pairs `embeddings[:, :-1]` with `action_embeddings[:, :-1]` against targets
`embeddings[:, 1:]`, so the predictor maps (state at t, action taken AT t) to
the state at t+1. Seeding with C real states therefore needs the C-1 actions
that produced states 1..C-1, followed by the actions to plan.
"""

from __future__ import annotations

import torch as t


def rollout_from_context(
    model,
    states: t.Tensor,
    actions: t.Tensor,
    max_context: int | None = None,
) -> t.Tensor:
    """[B, C, D] real embeddings + [B, C-1+H, 6] actions -> [B, H, D].

    The first C-1 actions are the ones already taken, one per real state after
    the first; the rest are the horizon to predict. With C=1 this is exactly
    `JEPA.predict_rollout`.
    """
    if states.ndim != 3:
        raise ValueError(f"states must be [B, C, D], got {tuple(states.shape)}")
    context_length = states.size(1)
    horizon = actions.size(1) - (context_length - 1)
    if horizon < 1:
        raise ValueError(
            f"{actions.size(1)} actions leave no horizon for {context_length} "
            "states of context"
        )

    limit = model.predictor.pos.size(1) if max_context is None else max_context
    conditions = model.action_encoder(actions)

    grown = states
    for step in range(horizon):
        # Index of the newest state, in `actions`/`conditions` coordinates.
        newest = context_length - 1 + step
        context = min(grown.size(1), limit)
        predictions = model.predictor(
            grown[:, -context:],
            conditions[:, newest + 1 - context : newest + 1],
        )
        grown = t.cat((grown, predictions[:, -1:]), dim=1)
    return grown[:, context_length:]
