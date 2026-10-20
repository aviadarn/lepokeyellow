"""A coordinate head trained WITH the model, not fitted onto it afterwards.

Why there has to be one. Measured on held-out episodes
(results/rollout_tiles.json), a read-out fitted on the frozen encoder places
the player from a real screen to 0.22 tiles, from the predictor's one-step
output to 2.42 -- and to 0.69 if it ignores the prediction entirely and assumes
the player did not move. The predictor is worse than standing still at saying
where the player will be, so every planner built on it was choosing between
numbers that carried no positional signal, and no read-out fitted after the
fact can recover what the predictor discarded. The one-step embedding MSE is
excellent at the same time, which is the point: nothing in `L_pred + lambda
L_reg` asks the predicted embedding to keep the player's tile legible.

So ask. This head reads (y, x) from an embedding, and it is trained on both the
encoder's real embeddings and the predictor's predicted ones against the tile
the emulator recorded. The gradient through the predicted ones is the part that
matters: it makes "where the player is" something the prediction has to carry.

One head for every map, deliberately. The coordinate frames are per-map and a
tile means nothing without its room, but the embedding already separates rooms
to 98.6%, so a shared head can condition on that -- and a per-map head would
need the map, which at planning time is itself a prediction.
"""

from __future__ import annotations

import torch as t
import torch.nn as nn


class CoordHead(nn.Module):
    def __init__(self, dim: int = 192, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, embeddings: t.Tensor) -> t.Tensor:
        """[..., D] -> [..., 2], in tiles."""
        return self.net(embeddings)


def coord_loss(
    head: CoordHead,
    predictions: t.Tensor,
    targets: t.Tensor,
    location: t.Tensor,
) -> tuple[t.Tensor, t.Tensor]:
    """Loss, and the predicted tile error to report.

    `location` is [B, T, 3] as recorded -- (map, y, x) with -1 for a read that
    failed, which is masked out rather than trained on. `predictions` and
    `targets` are aligned with `location[:, 1:]`, matching how `JEPA.forward`
    pairs `embeddings[:, :-1]` with `embeddings[:, 1:]`.

    The head is grounded on the real embeddings as well as the predicted ones.
    Without that, head and predictor could satisfy the loss together by
    agreeing on something that is not the player's position.
    """
    truth = location[:, 1:, 1:].float()
    mask = location[:, 1:, 0] >= 0
    if not bool(mask.any()):
        zero = predictions.new_zeros(())
        return zero, zero

    predicted = head(predictions)[mask]
    observed = head(targets)[mask]
    wanted = truth[mask]

    loss = (t.nn.functional.smooth_l1_loss(predicted, wanted)
            + t.nn.functional.smooth_l1_loss(observed, wanted))
    with t.no_grad():
        tiles = (predicted - wanted).abs().sum(dim=-1).mean()
    return loss, tiles
