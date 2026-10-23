"""Clone the oracle's Part 2 demonstrations into the policy.

TRAINING ONLY to produce; the result plays under the contract.

The point is not the training loss. 208 demonstrations against a 597k
parameter network will fit; what matters is whether a policy that has seen
the route walks it under the contract from Route 1, which is what
`scripts/eval_part2.py` measures. The RL run's failure was that it never
discovered the route at all -- seven of its eleven eval runs walked north
while the parcel is south -- so a clone is the cheapest way to find out
whether knowing the route is the missing piece or only part of it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from poke.rl.clone import action_index, save_clone, train_clone
from poke.rl.features import FrozenEncoder
from poke.goals import default_device


def load_episodes(path):
    """[(frames, action indices, objective key)] from a demonstration file."""
    episodes = []
    with h5py.File(path, "r") as file:
        spans = file["episodes"][:]
        meta = file["episode_meta"][:]
        frames = file["frames"]
        actions = file["actions"]
        for (start, length), row in zip(spans, meta):
            key = row["name"].decode() if isinstance(row["name"], bytes) else row["name"]
            chunk = frames[start:start + length, :, :, 0]
            labels = [action_index(a) for a in actions[start:start + length]]
            episodes.append((np.asarray(chunk), labels, key))
    return episodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demos", default="episodes/demos_part2.h5")
    parser.add_argument("--encoder", default="checkpoints/part2/best.pt")
    parser.add_argument("--out", default="checkpoints/rl/clone.pt")
    parser.add_argument("--report", default="results/clone_part2.json")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu",
                        help="cpu by default: 208 sequences of a 12.5M encoder "
                             "is dominated by per-call latency, not throughput")
    parser.add_argument("--half", action="store_true",
                        help="match a training run that used an fp16 encoder")
    args = parser.parse_args()

    encoder = FrozenEncoder(args.encoder, device=args.device, half=args.half)
    episodes = load_episodes(args.demos)
    counts = {key: len(actions) for _, actions, key in episodes}
    print(f"{sum(counts.values())} demonstrations: {counts}", flush=True)

    policy, history = train_clone(encoder, episodes, epochs=args.epochs,
                                  lr=args.lr, seed=args.seed,
                                  log_every=max(args.epochs // 10, 1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_clone(args.out, policy, encoder.dim, encoder.num_frames,
               demos=sum(counts.values()), history=history)

    # Training accuracy, stated for what it is: the clone reproducing the
    # demonstrations it was fitted on, not evidence it can play.
    correct = total = 0
    from poke.rl.clone import episode_observations
    with torch.no_grad():
        for frames, actions, key in episodes:
            obs, targets = episode_observations(encoder, frames, actions, key,
                                                policy.n_actions)
            hidden = policy.initial_hidden(1, "cpu")
            for step in range(len(targets)):
                logits, _, hidden = policy(torch.from_numpy(obs[step]).unsqueeze(0), hidden)
                correct += int(logits.argmax(-1).item() == int(targets[step]))
                total += 1

    report = {
        "demos": sum(counts.values()),
        "per_objective": counts,
        "epochs": args.epochs,
        "lr": args.lr,
        "first_loss": round(history[0], 4),
        "final_loss": round(history[-1], 4),
        "train_accuracy": round(correct / max(total, 1), 4),
        "encoder": args.encoder,
        "half": args.half,
        "caveat": "Training accuracy on 208 pairs is not evidence of play. "
                  "One optimal trajectory per objective, no recovery behaviour; "
                  "scripts/eval_part2.py under the contract is the measurement.",
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
