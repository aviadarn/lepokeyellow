"""Behaviour cloning the oracle's Part 2 demonstrations."""

import numpy as np
import pytest
import torch

from poke.actions import ACTIONS, BUTTON_ORDER, to_multihot
from poke.rl.clone import action_index, episode_observations, train_clone


def test_a_multihot_round_trips_to_its_action_index():
    for name in BUTTON_ORDER:
        index = ACTIONS.index(name)
        assert action_index(to_multihot(index)) == index


def test_an_empty_multihot_is_noop():
    """to_multihot(noop) sets no button, which is the honest encoding: noop
    is the absence of a press, not a seventh button."""
    assert action_index(np.zeros(len(BUTTON_ORDER), dtype=np.float32)) == ACTIONS.index("noop")


class FakeEncoder:
    dim, num_frames = 4, 2
    out_dim = 8

    def embed(self, frames):
        # One distinct value per frame, so a test can see the window roll.
        return np.array([[float(f[0, 0])] * self.dim for f in frames], dtype=np.float32)


def test_the_observation_is_built_the_way_the_policy_sees_it_at_play_time():
    """Same window, same objective one-hot, same previous action. A clone
    trained on a different observation than it is evaluated with learns
    nothing transferable."""
    frames = np.stack([np.full((144, 160), v, dtype=np.uint8) for v in (1, 2, 3)])
    actions = [ACTIONS.index("up"), ACTIONS.index("down"), ACTIONS.index("a")]
    obs, targets = episode_observations(FakeEncoder(), frames, actions,
                                        objective_key="get_parcel", n_actions=7)
    assert obs.shape == (3, 2 * 4 + 5 + 7)
    assert targets.tolist() == actions
    # First step: the window is the first frame tiled, previous action is noop.
    assert obs[0, :8].tolist() == [1.0] * 8
    assert obs[0, 8 + 2] == 1.0, "get_parcel is objective index 2"
    assert obs[0, 13 + ACTIONS.index("noop")] == 1.0
    # Second step: the window has rolled and the previous action is `up`.
    assert obs[1, :8].tolist() == [1.0] * 4 + [2.0] * 4
    assert obs[1, 13 + ACTIONS.index("up")] == 1.0


def test_training_drives_the_loss_down_on_its_own_demonstrations():
    """The set is 208 pairs and the net is 597k parameters, so fitting them
    is not evidence of anything except that the loop works."""
    torch.manual_seed(0)
    frames = np.stack([np.full((144, 160), v, dtype=np.uint8) for v in range(1, 9)])
    actions = [ACTIONS.index("up")] * 4 + [ACTIONS.index("right")] * 4
    episodes = [(frames, actions, "get_parcel")]
    policy, history = train_clone(FakeEncoder(), episodes, epochs=60, lr=1e-2, seed=0)
    assert history[-1] < history[0] / 2, f"loss went {history[0]:.3f} -> {history[-1]:.3f}"


def test_the_clone_saves_a_checkpoint_the_evaluator_can_load():
    """scripts/eval_part2.py reads blob["policy"] into an ActorCritic."""
    from poke.rl.clone import save_clone
    from poke.rl.policy import ActorCritic

    policy = ActorCritic(feature_dim=4, num_frames=2, hidden=8)
    blob = save_clone(None, policy, feature_dim=4, num_frames=2, demos=3)
    fresh = ActorCritic(feature_dim=4, num_frames=2, hidden=8)
    fresh.load_state_dict(blob["policy"])
    assert blob["feature_dim"] == 4 and blob["num_frames"] == 2


def test_context_steps_carry_the_sequence_but_are_not_scored():
    """DAgger's relabelled episodes are warmed with the frames the learner
    actually passed through. Those turns have the learner's own actions on
    them, not the expert's, so they must move the GRU and the frame window
    and contribute nothing to the loss."""
    frames = np.stack([np.full((144, 160), v, dtype=np.uint8) for v in (1, 2, 3, 4)])
    actions = [ACTIONS.index("a")] * 4
    policy, history = train_clone(FakeEncoder(), [(frames, actions, "get_parcel", 4)],
                                  epochs=3, lr=1e-2, seed=0)
    assert history == [0.0, 0.0, 0.0], "nothing scored, nothing learned"


def test_a_warmed_episode_sees_the_window_the_learner_had():
    """At the first scored step the window must hold the learner's preceding
    frames, not the expert's first frame tiled four times."""
    frames = np.stack([np.full((144, 160), v, dtype=np.uint8) for v in (1, 2, 3)])
    actions = [ACTIONS.index("left"), ACTIONS.index("right"), ACTIONS.index("up")]
    obs, _ = episode_observations(FakeEncoder(), frames, actions,
                                  objective_key="get_parcel", n_actions=7)
    # FakeEncoder has num_frames=2: step 2's window is frames 2 and 3.
    assert obs[2, :8].tolist() == [2.0] * 4 + [3.0] * 4
    assert obs[2, 13 + ACTIONS.index("right")] == 1.0, "the learner's own last press"
