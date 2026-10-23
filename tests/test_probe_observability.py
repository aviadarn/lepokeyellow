"""Is the expert's action a function of what the policy is allowed to see?

Two DAgger rounds relabelled the clone's own states and it still reached
nothing. Before adding a third, the question worth asking is whether the
observation identifies the state at all: the frozen encoder was measured
recovering which ROOM a frame is in (93.6%) and where in the room worse than
chance, and Route 1 is one room. If two tiles that need different buttons look
the same to the encoder, no amount of labelling makes the mapping learnable
and the clone can only memorise the order it saw them in.
"""

import numpy as np

from scripts.probe_observability import conflicts, loo_nn_accuracy


def test_a_feature_that_separates_the_labels_scores_one():
    features = np.array([[0.0], [0.1], [5.0], [5.1]])
    labels = np.array([0, 0, 1, 1])
    assert loo_nn_accuracy(features, labels) == 1.0


def test_a_feature_blind_to_the_labels_scores_nothing():
    """Four observations on top of each other, four different buttons: no
    neighbour can be right, whichever way the tie breaks."""
    features = np.zeros((4, 1))
    labels = np.array([0, 1, 2, 3])
    assert loo_nn_accuracy(features, labels) == 0.0


def test_a_point_is_never_its_own_neighbour():
    features = np.array([[0.0], [1.0]])
    labels = np.array([0, 1])
    assert loo_nn_accuracy(features, labels) == 0.0, "leave-one-out, or everything is 1.0"


def test_conflicts_counts_identical_observations_wanting_different_buttons():
    features = np.array([[0.0], [0.0], [1.0]])
    labels = np.array([0, 3, 3])
    found = conflicts(features, labels, tolerance=1e-6)
    assert found["groups"] == 1 and found["conflicting_steps"] == 2


def test_nothing_conflicts_when_every_observation_is_distinct():
    features = np.array([[0.0], [1.0], [2.0]])
    labels = np.array([0, 1, 2])
    assert conflicts(features, labels, tolerance=1e-6)["conflicting_steps"] == 0
