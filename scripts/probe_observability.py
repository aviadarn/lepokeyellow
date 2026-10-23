"""Does the observation identify the action the expert wants?

Behaviour cloning fits, DAgger relabels, and the clone still reaches nothing.
Both of those assume the thing being learned is a function: that two states
needing different buttons look different to the policy. This script checks
that assumption on the demonstrations themselves, with no emulator and no
training.

The measurement is a leave-one-out nearest neighbour over the observations a
contract-bound policy actually gets -- the frozen encoder's window plus the
objective -- against the same thing computed from the TRUE position. If the
expert's button is predictable from where the player is and not from what the
encoder sees, then the ceiling on any memoryless policy over these features is
the first number, and everything above it the clone reaches at training time
is the GRU memorising the order of a fixed trajectory.

1-NN and not a trained classifier on purpose: it has no capacity argument to
make. If the nearest observation in the whole dataset carries a different
button, no model reads the button off that observation.

TRAINING-TIME ANALYSIS. It reads the recorded locations, which play time
cannot.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from poke.actions import ACTIONS
from poke.goals import default_device
from poke.rl.clone import objective_one_hot
from poke.rl.features import EmbeddingStack, FrozenEncoder


def loo_nn_accuracy(features, labels) -> float:
    """Share of steps whose nearest OTHER step carries the same label."""
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    if len(labels) < 2:
        return 0.0
    square = (features ** 2).sum(axis=1)
    distance = square[:, None] + square[None, :] - 2 * features @ features.T
    np.fill_diagonal(distance, np.inf)
    return float((labels[distance.argmin(axis=1)] == labels).mean())


def conflicts(features, labels, tolerance: float = 1e-6) -> dict:
    """Groups of observations that are the same and want different buttons.

    The honest version of "the encoder cannot see the difference": not a
    distance that is small, but one below the tolerance, where the policy is
    being asked to return two answers for one input.
    """
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    groups: dict[bytes, list[int]] = defaultdict(list)
    for index, row in enumerate(features):
        groups[np.round(row / max(tolerance, 1e-12)).astype(np.int64).tobytes()].append(index)
    conflicting = [members for members in groups.values()
                   if len(set(labels[members].tolist())) > 1]
    return {
        "groups": len(conflicting),
        "conflicting_steps": sum(len(members) for members in conflicting),
        "largest_group": max((len(m) for m in conflicting), default=0),
    }


def episode_features(encoder, frames, objective_key):
    """The window the policy sees, per step, plus the objective one-hot.

    The previous action is deliberately left out. It is part of the play-time
    observation, but it is also the policy's own output, so including it would
    let a trajectory be identified by the sequence that produced it -- which is
    the memorisation this script exists to separate out.
    """
    embeddings = encoder.embed(frames)
    stack = EmbeddingStack(encoder.num_frames, encoder.dim)
    one_hot = objective_one_hot(objective_key)
    rows = []
    for step, embedding in enumerate(embeddings):
        if step == 0:
            stack.reset(embedding)
        else:
            stack.push(embedding)
        rows.append(np.concatenate([stack.vector(), np.asarray(one_hot, dtype=np.float32)]))
    return np.stack(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demos", nargs="+", default=["episodes/demos_part2.h5"])
    parser.add_argument("--encoder", default="checkpoints/part2/best.pt")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--out", default="results/observability_part2.json")
    args = parser.parse_args()

    import h5py

    from scripts.train_clone import load_episodes

    encoder = FrozenEncoder(args.encoder, device=args.device)
    seen, labels, locations, objectives = [], [], [], []
    for path in args.demos:
        episodes = load_episodes(path)
        with h5py.File(path, "r") as file:
            spans = file["episodes"][:]
            recorded = file["location"][:]
        for (start, length), (frames, actions, key, context) in zip(spans, episodes):
            features = episode_features(encoder, frames, key)
            for step in range(context, length):
                seen.append(features[step])
                labels.append(actions[step])
                locations.append(recorded[start + step])
                objectives.append(key)

    seen = np.stack(seen)
    labels = np.asarray(labels)
    locations = np.asarray(locations, dtype=np.float64)
    one_hots = np.stack([objective_one_hot(key) for key in objectives])
    # The same tile under two objectives is two states, on both sides.
    position = np.concatenate([locations * 1000.0, one_hots], axis=1)

    readable = [index for index, row in enumerate(locations) if (row >= 0).all()]
    report = {
        "demos": args.demos,
        "steps": int(len(labels)),
        "steps_with_a_recorded_position": len(readable),
        "label_counts": {ACTIONS[int(a)]: int(n) for a, n in Counter(labels.tolist()).items()},
        "majority_baseline": round(float(Counter(labels.tolist()).most_common(1)[0][1]
                                         / len(labels)), 4),
        "from_the_encoder": round(loo_nn_accuracy(seen, labels), 4),
        "from_the_true_position": round(
            loo_nn_accuracy(position[readable], labels[readable]), 4),
        "encoder_conflicts": conflicts(seen, labels, tolerance=1e-4),
        "position_conflicts": conflicts(position[readable], labels[readable], tolerance=1e-6),
        "what": "Leave-one-out 1-NN: does the nearest other observation in the "
                "whole dataset want the same button?",
        "why": "Cloning and DAgger both assume the mapping is a function. "
               "The encoder was measured recovering which room a frame is in "
               "and not where in the room; Route 1 is one room.",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
