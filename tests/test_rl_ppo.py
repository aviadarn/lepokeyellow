"""Hand-rolled PPO. The design asks for no heavy new dependency."""

import pytest
import torch

from poke.rl.policy import ActorCritic
from poke.rl.ppo import PPOConfig, RolloutBuffer, ppo_update, surrogate


def filled_buffer(rewards, values, dones, obs_dim=4):
    steps = len(rewards)
    buffer = RolloutBuffer(steps=steps, envs=1, obs_dim=obs_dim, device="cpu")
    for t in range(steps):
        buffer.add(t, obs=torch.zeros(1, obs_dim), action=torch.zeros(1, dtype=torch.long),
                   logprob=torch.zeros(1), value=torch.tensor([values[t]]),
                   reward=torch.tensor([rewards[t]]), done=torch.tensor([dones[t]]))
    return buffer


def test_gae_on_a_hand_computed_case():
    buffer = filled_buffer(rewards=[0.0, 0.0, 1.0], values=[0.0, 0.0, 0.0], dones=[0, 0, 0])
    config = PPOConfig(gamma=0.5, gae_lambda=1.0)
    buffer.compute_returns(last_value=torch.zeros(1), last_done=torch.zeros(1), config=config)
    assert buffer.advantages.squeeze(1).tolist() == pytest.approx([0.25, 0.5, 1.0])


def test_returns_are_advantages_plus_values():
    buffer = filled_buffer(rewards=[1.0, 1.0], values=[0.3, 0.7], dones=[0, 0])
    config = PPOConfig(gamma=0.9, gae_lambda=0.95)
    buffer.compute_returns(last_value=torch.zeros(1), last_done=torch.zeros(1), config=config)
    assert torch.allclose(buffer.returns, buffer.advantages + buffer.values)


def test_a_done_stops_the_bootstrap():
    """Reward after a terminal step belongs to the next episode, not this one."""
    buffer = filled_buffer(rewards=[0.0, 5.0], values=[0.0, 0.0], dones=[1, 0])
    config = PPOConfig(gamma=1.0, gae_lambda=1.0)
    buffer.compute_returns(last_value=torch.zeros(1), last_done=torch.zeros(1), config=config)
    assert buffer.advantages[0].item() == pytest.approx(0.0)
    assert buffer.advantages[1].item() == pytest.approx(5.0)


def test_the_surrogate_clips_a_runaway_ratio():
    loss = surrogate(ratio=torch.tensor([5.0]), advantage=torch.tensor([1.0]), clip=0.2)
    assert loss.item() == pytest.approx(-1.2)


def test_the_surrogate_does_not_clip_a_ratio_inside_the_band():
    loss = surrogate(ratio=torch.tensor([1.1]), advantage=torch.tensor([1.0]), clip=0.2)
    assert loss.item() == pytest.approx(-1.1)


def test_a_negative_advantage_clips_on_the_other_side():
    loss = surrogate(ratio=torch.tensor([0.1]), advantage=torch.tensor([-1.0]), clip=0.2)
    assert loss.item() == pytest.approx(0.8)


def test_an_update_makes_the_rewarded_action_more_likely():
    """Action 0 pays, action 1 does not.

    Rewarding every step equally would teach nothing: advantage normalisation
    turns a constant advantage into zero, correctly -- "all actions were
    equally good" is what the data says -- and the entropy bonus is then the
    only gradient left, which pushes towards uniform.
    """
    torch.manual_seed(0)
    policy = ActorCritic(feature_dim=4, num_frames=1, hidden=8)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.05)
    steps, envs = 8, 2
    buffer = RolloutBuffer(steps=steps, envs=envs, obs_dim=policy.obs_dim, device="cpu")
    obs = torch.zeros(envs, policy.obs_dim)
    hidden = policy.initial_hidden(envs, "cpu")
    with torch.no_grad():
        for t in range(steps):
            action = torch.full((envs,), t % 2, dtype=torch.long)
            logits, value, hidden = policy(obs, hidden)
            logprob = torch.log_softmax(logits, -1).gather(1, action.unsqueeze(1)).squeeze(1)
            buffer.add(t, obs=obs, action=action, logprob=logprob, value=value,
                       reward=(1.0 - action).float(), done=torch.zeros(envs))
    config = PPOConfig(epochs=4, minibatch_envs=2)
    buffer.compute_returns(torch.zeros(envs), torch.zeros(envs), config)
    before = torch.softmax(policy(obs, policy.initial_hidden(envs, "cpu"))[0], -1)[0, 0]
    stats = ppo_update(policy, optimizer, buffer, config, policy.initial_hidden(envs, "cpu"))
    after = torch.softmax(policy(obs, policy.initial_hidden(envs, "cpu"))[0], -1)[0, 0]
    assert after > before
    assert set(stats) >= {"policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction"}


def test_an_update_reports_a_finite_kl():
    torch.manual_seed(0)
    policy = ActorCritic(feature_dim=4, num_frames=1, hidden=8)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    buffer = RolloutBuffer(steps=4, envs=2, obs_dim=policy.obs_dim, device="cpu")
    for t in range(4):
        buffer.add(t, obs=torch.zeros(2, policy.obs_dim),
                   action=torch.zeros(2, dtype=torch.long), logprob=torch.zeros(2),
                   value=torch.zeros(2), reward=torch.zeros(2), done=torch.zeros(2))
    config = PPOConfig()
    buffer.compute_returns(torch.zeros(2), torch.zeros(2), config)
    stats = ppo_update(policy, optimizer, buffer, config, policy.initial_hidden(2, "cpu"))
    assert torch.isfinite(torch.tensor(stats["approx_kl"]))


def test_minibatches_split_by_env_not_by_time():
    """A GRU minibatch has to keep whole trajectories, or the hidden state it
    replays from belongs to a different episode."""
    config = PPOConfig(minibatch_envs=2)
    assert list(RolloutBuffer.env_batches(envs=4, size=config.minibatch_envs)) == [[0, 1], [2, 3]]


def _buffer(rewards, dones, obs_dim=4):
    """A one-env buffer with zero values, so GAE reduces to discounted sums."""
    buffer = RolloutBuffer(steps=len(rewards), envs=1, obs_dim=obs_dim)
    for t, (reward, done) in enumerate(zip(rewards, dones)):
        buffer.add(t, torch.zeros(1, obs_dim), torch.zeros(1, dtype=torch.long),
                   torch.zeros(1), torch.zeros(1),
                   torch.tensor([reward]), torch.tensor([done]))
    return buffer


def test_the_reward_that_ends_the_episode_is_not_discarded():
    """TrainEnv ends the episode on the step that pays enter_forest's +1.
    `alive = 1 - dones[t]` zeroed exactly that step, so the one reward the
    whole run is chasing produced no advantage and no return target."""
    config = PPOConfig(gamma=1.0, gae_lambda=1.0)
    buffer = _buffer(rewards=[0.0, 0.0, 1.0], dones=[0.0, 0.0, 1.0])
    buffer.compute_returns(torch.zeros(1), torch.ones(1), config)
    assert buffer.advantages[2].item() == pytest.approx(1.0), "the goal step pays"
    assert buffer.returns[0].item() == pytest.approx(1.0), "and it reaches step 0"


def test_an_episode_boundary_does_not_leak_reward_backwards():
    """Two episodes in one rollout: the second's reward must not be credited
    to actions taken in the first."""
    config = PPOConfig(gamma=1.0, gae_lambda=1.0)
    buffer = _buffer(rewards=[0.0, 1.0, 0.0, 2.0], dones=[0.0, 1.0, 0.0, 0.0])
    buffer.compute_returns(torch.zeros(1), torch.zeros(1), config)
    assert [round(v, 5) for v in buffer.advantages[:, 0].tolist()] == [1.0, 1.0, 2.0, 2.0]


def test_evaluate_forgets_the_previous_episode_at_a_done():
    """The driver clears a column's GRU row when its episode ends. evaluate()
    replays the rollout and must clear it at the same place, or the replayed
    hidden state is one that never existed."""
    policy = ActorCritic(feature_dim=4, num_frames=1, hidden=8)
    obs = torch.randn(3, 1, policy.obs_dim)
    actions = torch.zeros(3, 1, dtype=torch.long)
    dones = torch.tensor([[1.0], [0.0], [0.0]])
    one = policy.evaluate(obs, torch.ones(1, 8), actions, dones=dones)
    other = policy.evaluate(obs, torch.full((1, 8), -3.0), actions, dones=dones)
    assert not torch.allclose(one["logprobs"][0], other["logprobs"][0]), \
        "before the boundary the two histories differ"
    assert torch.allclose(one["logprobs"][1:], other["logprobs"][1:]), \
        "after it they must not"


def test_the_update_skips_a_dropped_workers_column():
    """Review focus 5: a dead worker's column is blank frames, a zero
    objective one-hot and zero rewards. Training on it shifts the advantage
    normalisation for every live column."""
    seen = []
    policy = ActorCritic(feature_dim=4, num_frames=1, hidden=8)
    real_evaluate = policy.evaluate

    def spy(obs_seq, hidden0, actions, dones=None):
        seen.append(obs_seq.size(1))
        return real_evaluate(obs_seq, hidden0, actions, dones=dones)

    policy.evaluate = spy
    buffer = RolloutBuffer(steps=2, envs=4, obs_dim=policy.obs_dim)
    config = PPOConfig(epochs=1, minibatch_envs=2)
    ppo_update(policy, torch.optim.Adam(policy.parameters()), buffer, config,
               policy.initial_hidden(4, "cpu"), usable=[0, 1, 2])
    assert sum(seen) == 3, f"one epoch must cover exactly the 3 usable columns, saw {seen}"
