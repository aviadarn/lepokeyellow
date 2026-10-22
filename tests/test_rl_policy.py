"""The policy: four embeddings, the objective, the last button -> a button."""

import numpy as np
import torch

from poke.actions import n_actions
from poke.objectives import N_OBJECTIVES
from poke.rl.policy import ActorCritic, observation


def small() -> ActorCritic:
    return ActorCritic(feature_dim=8, num_frames=4, hidden=16)


def test_observation_is_window_then_objective_then_previous_action():
    window = np.arange(8, dtype=np.float32)
    obs = observation(window, [0.0, 0.0, 1.0, 0.0, 0.0], prev_action=2, n_actions=7)
    assert obs.shape == (8 + 5 + 7,)
    assert obs[:8].tolist() == list(range(8))
    assert obs[8:13].tolist() == [0, 0, 1, 0, 0]
    assert obs[13:].tolist() == [0, 0, 1, 0, 0, 0, 0]


def test_obs_dim_matches_what_observation_builds():
    policy = small()
    window = np.zeros(8 * 4, dtype=np.float32)
    obs = observation(window, [0.0] * N_OBJECTIVES, prev_action=0, n_actions=n_actions())
    assert obs.shape == (policy.obs_dim,)


def test_forward_returns_logits_value_and_the_next_hidden():
    policy = small()
    hidden = policy.initial_hidden(batch=3, device="cpu")
    obs = torch.zeros(3, policy.obs_dim)
    logits, value, nxt = policy(obs, hidden)
    assert logits.shape == (3, n_actions())
    assert value.shape == (3,)
    assert nxt.shape == hidden.shape


def test_the_hidden_state_actually_changes_the_output():
    policy = small()
    obs = torch.zeros(1, policy.obs_dim)
    zero = policy.initial_hidden(batch=1, device="cpu")
    warm = torch.ones_like(zero)
    assert not torch.allclose(policy(obs, zero)[0], policy(obs, warm)[0])


def test_greedy_takes_the_argmax():
    policy = small()
    with torch.no_grad():
        policy.actor.bias.copy_(torch.tensor([0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 0.0]))
        policy.actor.weight.zero_()
    out = policy.act(torch.zeros(1, policy.obs_dim),
                     policy.initial_hidden(1, "cpu"), greedy=True)
    assert out["action"].tolist() == [2]


def test_act_reports_a_logprob_for_the_action_it_chose():
    policy = small()
    out = policy.act(torch.zeros(2, policy.obs_dim), policy.initial_hidden(2, "cpu"))
    logits, _, _ = policy(torch.zeros(2, policy.obs_dim), policy.initial_hidden(2, "cpu"))
    expected = torch.log_softmax(logits, dim=-1).gather(1, out["action"].unsqueeze(1)).squeeze(1)
    assert torch.allclose(out["logprob"], expected, atol=1e-6)


def test_evaluate_replays_a_sequence_through_the_gru():
    policy = small()
    steps, batch = 5, 2
    obs = torch.zeros(steps, batch, policy.obs_dim)
    actions = torch.zeros(steps, batch, dtype=torch.long)
    out = policy.evaluate(obs, policy.initial_hidden(batch, "cpu"), actions)
    assert out["logprobs"].shape == (steps, batch)
    assert out["values"].shape == (steps, batch)
    assert out["entropy"].shape == (steps, batch)


def test_evaluate_agrees_with_act_on_the_first_step():
    policy = small()
    obs = torch.randn(1, 1, policy.obs_dim)
    hidden = policy.initial_hidden(1, "cpu")
    stepwise = policy.act(obs[0], hidden, greedy=True)
    replayed = policy.evaluate(obs, hidden, stepwise["action"].unsqueeze(0))
    assert torch.allclose(replayed["logprobs"][0], stepwise["logprob"], atol=1e-6)


def test_the_real_policy_is_under_a_million_parameters():
    policy = ActorCritic()
    total = sum(p.numel() for p in policy.parameters())
    assert total < 1_000_000, f"{total} parameters; the design says well under 1M"
