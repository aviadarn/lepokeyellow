"""The world-model exploration bonus. Training only.

Plan2Explore-style: the policy is paid for transitions the world model did not
see coming.

Two rules the design is explicit about, both kept by the caller rather than
here: the bonus is computed only on transitions the policy controlled, because
text boxes and the Pikachu-follow sprite are noisy-TV sources that a handler
turn walks straight into; and it is never a policy input, only a reward term.

Scale: raw latent MSE has no natural size, and an objective pays exactly 1.0.
So the raw value is divided by its own running standard deviation before the
coefficient applies, which keeps a surprising frame worth a few hundredths.

That division needs a spread to divide by. Measured here while building it:
the first transition has none -- count 1, std 0 -- and dividing by the 1e-8
guard paid 2,500,000 for one frame, which would have drowned every objective
in the run. So the bonus pays nothing until WARMUP transitions have set a
scale, and a single outlier is clipped at CLIP standard deviations after that.
"""

from __future__ import annotations

import numpy as np
import torch

from poke.actions import to_multihot

#: Transitions to watch before paying anything, so there is a spread to
#: divide by.
WARMUP = 32
#: Standard deviations one transition may be worth. Without it a single
#: outlier -- a warp, a battle opening -- is worth more than a milestone.
CLIP = 5.0

#: Which signal, and how it was picked. The design asks for a comparison
#: against ensemble disagreement; this arm skipped it, and that is recorded
#: here rather than left for a reader to discover from a commit message.
CHOSEN = {
    "signal": "prediction_error",
    "measured": False,
    "why": "one arm, and prediction error needs no second model -- the frozen "
           "JEPA already has a predictor. The comparison against ensemble "
           "disagreement is deferred to a later plan.",
    "alternative": "ensemble_disagreement",
}


class RunningNorm:
    """Welford mean/variance, for scaling a reward term of unknown size."""

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self._m2 = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self._m2 += delta * (value - self.mean)

    @property
    def std(self) -> float:
        return 0.0 if self.count < 2 else float(np.sqrt(self._m2 / self.count))

    def normalise(self, value: float) -> float:
        return float(value) / (self.std + 1e-8)


class WorldModelBonus:
    def __init__(self, jepa, device: str = "cpu", coef: float = 0.05):
        self.jepa = jepa
        self.device = device
        self.coef = coef
        self.norm = RunningNorm()

    def bonus(self, prev_embedding, action_index: int, next_embedding) -> float:
        """How wrong the predictor was about this transition, scaled."""
        return self.bonus_many(np.asarray(prev_embedding)[None, ...], [action_index],
                               np.asarray(next_embedding)[None, ...])[0]

    def _dtype(self):
        """The predictor's precision. It may be fp16 -- the encoder is frozen,
        so the driver is free to halve it, and then its inputs must follow."""
        try:
            return next(self.jepa.parameters()).dtype
        except (AttributeError, StopIteration):
            return torch.float32

    def bonus_many(self, prev_embeddings, action_indices, next_embeddings) -> list[float]:
        """One predictor forward for a whole rollout step.

        The driver pays every env on every step. Four separate forwards on MPS
        cost more wall-clock than the four emulators they describe -- measured
        at 60 driver steps/s against a 400 floor -- and the predictor batches
        over the leading dimension anyway.
        """
        dtype = self._dtype()
        initial = torch.from_numpy(np.asarray(prev_embeddings, dtype=np.float32)).to(dtype)
        buttons = torch.from_numpy(
            np.stack([to_multihot(index) for index in action_indices])).unsqueeze(1).to(dtype)
        with torch.no_grad():
            predicted = self.jepa.rollout(initial.to(self.device), buttons.to(self.device))[:, 0]
        target = torch.from_numpy(np.asarray(next_embeddings, dtype=np.float32))
        errors = ((predicted.float().cpu() - target) ** 2).mean(dim=-1).tolist()
        out = []
        for error in errors:
            self.norm.update(float(error))
            out.append(0.0 if self.norm.count < WARMUP
                       else self.coef * min(self.norm.normalise(float(error)), CLIP))
        return out
