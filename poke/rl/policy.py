"""The policy: a small recurrent actor-critic over frozen embeddings.

Input, in the design's words: four frame embeddings, the objective one-hot and
the previous action. Body: an MLP into a GRU for memory. Output: logits over
the seven existing actions and a value.

Memory is not decoration here. The contract forbids reading position, so "I
just walked into this wall" is only available to the policy as a difference
between successive embeddings, and the GRU is where that difference can live.

The previous action is in the observation for the same reason: a frame after a
no-op and a frame after walking into a wall are the same frame.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from poke.actions import n_actions as action_count
from poke.objectives import N_OBJECTIVES


def observation(window, objective_one_hot, prev_action: int, n_actions: int) -> np.ndarray:
    """The policy's input vector: embeddings, objective, previous action."""
    previous = np.zeros(n_actions, dtype=np.float32)
    previous[prev_action] = 1.0
    return np.concatenate([
        np.asarray(window, dtype=np.float32),
        np.asarray(objective_one_hot, dtype=np.float32),
        previous,
    ])


class ActorCritic(nn.Module):
    def __init__(
        self,
        feature_dim: int = 192,
        num_frames: int = 4,
        n_objectives: int = N_OBJECTIVES,
        n_actions: int | None = None,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        self.n_actions = n_actions or action_count()
        self.hidden_size = hidden
        self.obs_dim = feature_dim * num_frames + n_objectives + self.n_actions
        self.body = nn.Sequential(nn.Linear(self.obs_dim, hidden), nn.Tanh())
        self.gru = nn.GRUCell(hidden, hidden)
        self.actor = nn.Linear(hidden, self.n_actions)
        self.critic = nn.Linear(hidden, 1)
        # A near-uniform policy at step zero. Large initial logits make PPO's
        # first ratio enormous and the clip does the rest of the damage.
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)

    def initial_hidden(self, batch: int, device) -> torch.Tensor:
        return torch.zeros(batch, self.hidden_size, device=device)

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor):
        hidden = self.gru(self.body(obs), hidden)
        return self.actor(hidden), self.critic(hidden).squeeze(-1), hidden

    def act(self, obs: torch.Tensor, hidden: torch.Tensor, greedy: bool = False) -> dict:
        logits, value, hidden = self(obs, hidden)
        distribution = torch.distributions.Categorical(logits=logits)
        action = logits.argmax(dim=-1) if greedy else distribution.sample()
        return {
            "action": action,
            "logprob": distribution.log_prob(action),
            "value": value,
            "entropy": distribution.entropy(),
            "hidden": hidden,
        }

    def evaluate(self, obs_seq: torch.Tensor, hidden0: torch.Tensor,
                 actions: torch.Tensor) -> dict:
        """Replay [T, B, obs] through the GRU from a stored hidden state.

        PPO needs the logprobs of actions already taken, under the current
        weights. With a recurrent policy that means replaying the sequence in
        order -- shuffling timesteps into minibatches would feed the GRU a
        hidden state from a different trajectory.
        """
        logprobs, values, entropies = [], [], []
        hidden = hidden0
        for step in range(obs_seq.size(0)):
            logits, value, hidden = self(obs_seq[step], hidden)
            distribution = torch.distributions.Categorical(logits=logits)
            logprobs.append(distribution.log_prob(actions[step]))
            values.append(value)
            entropies.append(distribution.entropy())
        return {
            "logprobs": torch.stack(logprobs),
            "values": torch.stack(values),
            "entropy": torch.stack(entropies),
            "hidden": hidden,
        }
