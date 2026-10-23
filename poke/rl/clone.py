"""Behaviour cloning the oracle's Part 2 demonstrations.

TRAINING TIME. The demonstrations were found with emulator snapshots, which
the contract bans at play time; what the clone learns is frame -> button, and
it plays under the contract like any other policy.

The observation has to be assembled exactly the way `scripts/eval_part2.py`
assembles it, or the clone is trained on one input and evaluated on another:
a rolling window of frozen-encoder embeddings over the policy's OWN turns,
the objective one-hot, and the previous action. The demonstrations are
already filtered to policy turns, so "consecutive frames" means the same
thing on both sides.

What this cannot fix: 208 pairs is one optimal trajectory per objective with
no recovery behaviour. A clone that steps off the route has never seen a
frame showing the way back. Fitting the training set proves the loop works
and nothing else; the eval under the contract is the measurement.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from poke.actions import ACTIONS, BUTTON_ORDER, n_actions as action_count
from poke.objectives import CHAIN, N_OBJECTIVES
from poke.rl.features import EmbeddingStack
from poke.rl.policy import ActorCritic, observation

NOOP = ACTIONS.index("noop")


def action_index(multihot) -> int:
    """The action a recorded multihot stands for.

    `to_multihot` writes no bit for noop, so an all-zero row is noop rather
    than a missing label.
    """
    array = np.asarray(multihot)
    if not array.any():
        return NOOP
    return ACTIONS.index(BUTTON_ORDER[int(np.argmax(array))])


def objective_one_hot(key: str) -> list[float]:
    """Matches `poke.objectives.one_hot`, which indexes by position in CHAIN."""
    vector = [0.0] * N_OBJECTIVES
    for objective in CHAIN:
        if objective.key == key:
            vector[objective.index] = 1.0
            return vector
    raise ValueError(f"{key} is not an objective; known: "
                     + ", ".join(o.key for o in CHAIN))


def unpack(episode):
    """(frames, actions, objective key, context length) from an episode.

    Episodes from `scripts/collect_demos.py` are three-tuples and start at the
    beginning of their own trajectory. A DAgger episode carries a fourth
    element: how many leading steps are the learner's own turns, replayed to
    warm the GRU and the frame window and scored on nothing.
    """
    frames, actions, key, *rest = episode
    return frames, actions, key, int(rest[0]) if rest else 0


def episode_observations(encoder, frames, actions, objective_key: str,
                         n_actions: int | None = None):
    """[(T, obs_dim) observations, (T,) action targets] for one episode."""
    n_actions = n_actions or action_count()
    embeddings = encoder.embed(frames)
    stack = EmbeddingStack(encoder.num_frames, encoder.dim)
    one_hot = objective_one_hot(objective_key)
    rows, previous = [], NOOP
    for step, embedding in enumerate(embeddings):
        if step == 0:
            stack.reset(embedding)
        else:
            stack.push(embedding)
        rows.append(observation(stack.vector(), one_hot, previous, n_actions))
        previous = actions[step]
    return np.stack(rows).astype(np.float32), np.asarray(actions, dtype=np.int64)


def train_clone(encoder, episodes, epochs: int = 400, lr: float = 1e-3,
                hidden: int = 256, seed: int = 0, log_every: int = 0):
    """Cross-entropy on the demonstrated action, replayed through the GRU.

    One sequence per episode, in order: the policy is recurrent, and a
    shuffled batch of timesteps would feed it a hidden state from another
    trajectory -- the same rule the PPO minibatching follows.
    """
    torch.manual_seed(seed)
    policy = ActorCritic(feature_dim=encoder.dim, num_frames=encoder.num_frames,
                         hidden=hidden)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    prepared = []
    for episode in episodes:
        frames, actions, key, context = unpack(episode)
        obs, targets = episode_observations(encoder, frames, actions, key,
                                            policy.n_actions)
        prepared.append((torch.from_numpy(obs), torch.from_numpy(targets), context))

    history = []
    for epoch in range(epochs):
        total, steps = 0.0, 0
        for obs, targets, context in prepared:
            if context >= obs.size(0):
                continue
            hidden_state = policy.initial_hidden(1, obs.device)
            logits = []
            for step in range(obs.size(0)):
                logit, _, hidden_state = policy(obs[step].unsqueeze(0), hidden_state)
                logits.append(logit)
            loss = nn.functional.cross_entropy(torch.cat(logits)[context:],
                                               targets[context:])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
            total += float(loss.item())
            steps += 1
        history.append(total / max(steps, 1))
        if log_every and epoch % log_every == 0:
            print(f"epoch {epoch:4d} loss {history[-1]:.4f}", flush=True)
    return policy, history


def save_clone(path, policy, feature_dim: int, num_frames: int, demos: int,
               history=None) -> dict:
    """A blob in the shape scripts/eval_part2.py expects."""
    blob = {
        "policy": policy.state_dict(),
        "feature_dim": feature_dim,
        "num_frames": num_frames,
        "demos": demos,
        "source": "behaviour cloning on the oracle's Part 2 solution",
        "final_loss": None if not history else history[-1],
    }
    if path is not None:
        torch.save(blob, path)
    return blob
