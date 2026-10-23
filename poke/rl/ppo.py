"""PPO, written here rather than pulled in.

The design prefers no heavy new dependency, and the recurrent part is the
reason it is cheap to write and expensive to borrow: with a GRU in the policy,
a minibatch is a set of whole trajectories, not a shuffled bag of timesteps.
Most implementations shuffle. One that shuffles here feeds the GRU a hidden
state recorded in a different episode, and the logprobs it computes are of a
policy that never existed.

So: the buffer is [T, N], advantages are computed backwards per env with the
done mask breaking the chain, and minibatches are splits over N.

gamma is 0.999 by default. The eval budget is 1,461 actions and the parcel
errand is a few hundred of them; at the usual 0.99 a reward 400 steps away is
worth 0.018 of itself, which is indistinguishable from the tile bonus noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class PPOConfig:
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    epochs: int = 4
    #: Envs per minibatch. Whole trajectories, never sliced in time.
    minibatch_envs: int = 2


def surrogate(ratio: torch.Tensor, advantage: torch.Tensor, clip: float) -> torch.Tensor:
    """The clipped objective, as a loss (so: negated, and averaged)."""
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantage
    return -torch.min(unclipped, clipped).mean()


class RolloutBuffer:
    def __init__(self, steps: int, envs: int, obs_dim: int, device="cpu"):
        self.steps, self.envs = steps, envs
        self.device = device
        self.obs = torch.zeros(steps, envs, obs_dim, device=device)
        self.actions = torch.zeros(steps, envs, dtype=torch.long, device=device)
        self.logprobs = torch.zeros(steps, envs, device=device)
        self.values = torch.zeros(steps, envs, device=device)
        self.rewards = torch.zeros(steps, envs, device=device)
        self.dones = torch.zeros(steps, envs, device=device)
        self.advantages = torch.zeros(steps, envs, device=device)
        self.returns = torch.zeros(steps, envs, device=device)

    def add(self, t: int, obs, action, logprob, value, reward, done) -> None:
        self.obs[t] = obs
        self.actions[t] = action
        self.logprobs[t] = logprob
        self.values[t] = value
        self.rewards[t] = reward
        self.dones[t] = done

    def compute_returns(self, last_value, last_done, config: PPOConfig) -> None:
        """Generalised advantage estimation, backwards, done-masked.

        One convention throughout: `dones[t]` means *the action at t ended the
        episode*. So the mask belongs on what comes after t, never on t itself.
        An earlier version multiplied the whole term by `1 - dones[t]`, which
        zeroed the terminal step -- and the terminal step is where this task
        pays `enter_forest` its +1, so the goal produced no gradient at all.

        `last_done` is unused under this convention and kept for the signature:
        whether the rollout's final action ended its episode is already in
        `dones[-1]`.
        """
        advantage = torch.zeros(self.envs, device=self.device)
        for t in reversed(range(self.steps)):
            next_value = last_value if t == self.steps - 1 else self.values[t + 1]
            # Zero for a step that ended the episode: no bootstrap, no chain
            # across the boundary. The reward at t is still paid.
            nonterminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + config.gamma * next_value * nonterminal - self.values[t]
            advantage = delta + config.gamma * config.gae_lambda * nonterminal * advantage
            self.advantages[t] = advantage
        self.returns = self.advantages + self.values

    @staticmethod
    def env_batches(envs, size: int):
        """Split env columns into minibatches. `envs` is a count or an
        explicit list of column indices (a count means all of them)."""
        columns = list(range(envs)) if isinstance(envs, int) else list(envs)
        for start in range(0, len(columns), size):
            yield columns[start:start + size]


def ppo_update(policy, optimizer, buffer: RolloutBuffer, config: PPOConfig,
               hidden0: torch.Tensor, usable: list[int] | None = None) -> dict:
    """One PPO pass over a filled buffer. Returns the diagnostics worth logging.

    `hidden0` must be the hidden state the rollout was *generated* from. Pass a
    zero state and `evaluate` replays the sequence from a policy that never
    produced the data: the first epoch's importance ratio is not 1, so the
    trust region is measured against the wrong reference.

    `usable` names the env columns worth training on -- everything except
    workers that died mid-rollout, whose columns are blank frames and a zero
    objective one-hot. Those columns also have to stay out of the advantage
    normalisation, or they move the baseline for every live column.
    """
    columns = list(range(buffer.envs)) if usable is None else list(usable)
    if not columns:
        return {key: 0.0 for key in
                ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction")}
    live = torch.tensor(columns, device=buffer.device)
    advantages = buffer.advantages
    kept = advantages[:, live]
    normalised = (advantages - kept.mean()) / (kept.std() + 1e-8)

    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
             "approx_kl": 0.0, "clip_fraction": 0.0}
    passes = 0
    for _ in range(config.epochs):
        for envs in RolloutBuffer.env_batches(columns, config.minibatch_envs):
            index = torch.tensor(envs, device=buffer.device)
            out = policy.evaluate(buffer.obs[:, index], hidden0[index],
                                  buffer.actions[:, index], dones=buffer.dones[:, index])
            ratio = (out["logprobs"] - buffer.logprobs[:, index]).exp()
            policy_loss = surrogate(ratio, normalised[:, index], config.clip)
            value_loss = nn.functional.mse_loss(out["values"], buffer.returns[:, index])
            entropy = out["entropy"].mean()
            loss = policy_loss + config.vf_coef * value_loss - config.ent_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += entropy.item()
                stats["approx_kl"] += ((ratio - 1) - ratio.log()).mean().item()
                stats["clip_fraction"] += ((ratio - 1).abs() > config.clip).float().mean().item()
            passes += 1

    return {key: value / max(passes, 1) for key, value in stats.items()}
