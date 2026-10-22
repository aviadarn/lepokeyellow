"""Score the trained policy under the play-time contract.

Ten sampling seeds and one greedy run, each a single continuous episode from
states/oracle/route1.state with a 1,461-action budget -- 3x the oracle's 487,
fixed in results/part2_ceiling.json. The policy sees `PlayTimeEnv`: pixels, the
frozen encoder, the objective one-hot. Any reach for the map id or the
coordinates raises rather than returning a number.

Every run writes a GIF. The design is explicit that a human watches them,
because a reward hack shows up there before it shows up in a table.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from poke.contract import Run
from poke.env import PokeEnv
from poke.goals import default_device
from poke.objectives import CHAIN
from poke.rl.features import EmbeddingStack, FrozenEncoder
from poke.rl.policy import ActorCritic, observation
from scripts.play2 import play

#: 3x the oracle's 487 actions.
BUDGET = 1461
#: The design's bar: map 51 on at least this many of ten seeds.
DONE_SEEDS = 5


class NetPolicy:
    """The trained net, in the slot scripts/play2.py gives a policy.

    It touches exactly one thing on the guarded env: `screen()`. The frame
    history and the GRU state live here, because the contract gives the policy
    no memory of its own.
    """

    def __init__(self, policy, encoder, device="cpu", greedy=False, seed=0):
        self.policy = policy
        self.encoder = encoder
        self.device = device
        self.greedy = greedy
        self.rng = random.Random(seed)
        torch.manual_seed(seed)
        self.stack = EmbeddingStack(encoder.num_frames, encoder.dim)
        self.reset()

    def reset(self) -> None:
        self.hidden = self.policy.initial_hidden(1, self.device)
        self.prev_action = 0
        self.stack.buffer = None
        self.frames: list[np.ndarray] = []

    def __call__(self, env, objective_one_hot) -> int:
        screen = env.screen()
        self.frames.append(screen)
        embedding = self.encoder.embed(screen[:, :, 0][None, ...])[0]
        if self.stack.buffer is None:
            self.stack.reset(embedding)
        else:
            self.stack.push(embedding)
        obs = observation(self.stack.vector(), objective_one_hot,
                          self.prev_action, self.policy.n_actions)
        tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.policy.act(tensor, self.hidden, greedy=self.greedy)
        self.hidden = out["hidden"]
        self.prev_action = int(out["action"].item())
        return self.prev_action


def write_gif(frames, path, every=6, scale=1) -> None:
    """Subsampled, so a 1,461-action run is a few hundred kilobytes."""
    if not frames:
        return
    images = [Image.fromarray(frame) for frame in frames[::every]]
    if scale != 1:
        images = [img.resize((img.width * scale, img.height * scale), Image.NEAREST)
                  for img in images]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    images[0].save(path, save_all=True, append_images=images[1:],
                   duration=60, loop=0, optimize=True)


def summarise(runs) -> dict:
    keys = [objective.key for objective in CHAIN]
    reached = {key: sum(1 for run in runs if run["objectives"].get(key) is not None)
               for key in keys}
    shares = [run["learned_share"] for run in runs]
    forest_runs = [run for run in runs if run["objectives"].get("enter_forest") is not None]
    return {
        "seeds": len(runs),
        "reached": reached,
        "passes_done_threshold": reached.get("enter_forest", 0) >= DONE_SEEDS,
        "median_learned_share": statistics.median(shares) if shares else 0.0,
        "actions_to_forest": sorted(run["objectives"]["enter_forest"] for run in forest_runs),
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rom", default="pokeyellow.gbc")
    parser.add_argument("--state", default="states/oracle/route1.state")
    parser.add_argument("--encoder", default="checkpoints/part2/best.pt")
    parser.add_argument("--policy", default="checkpoints/rl/last.pt")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--gif-dir", default="media")
    parser.add_argument("--out", default="results/part2_rl_eval.json")
    args = parser.parse_args()

    encoder = FrozenEncoder(args.encoder, device=args.device)
    net = ActorCritic(feature_dim=encoder.dim, num_frames=encoder.num_frames).to(args.device)
    blob = torch.load(args.policy, map_location="cpu", weights_only=False)
    net.load_state_dict(blob["policy"])
    net.eval()

    runs = []
    for seed in list(range(args.seeds)) + ["greedy"]:
        greedy = seed == "greedy"
        policy = NetPolicy(net, encoder, device=args.device, greedy=greedy,
                           seed=0 if greedy else seed)
        env = PokeEnv(args.rom, args.state)
        env.reset()
        # No settle(): it shifts the frame phase, measured in Stage 1.
        result = play(env, policy, budget=args.budget, run=Run(env))
        result["seed"] = seed
        result["greedy"] = greedy
        gif = Path(args.gif_dir) / f"part2_rl_{seed}.gif"
        write_gif(policy.frames, gif)
        result["gif"] = str(gif)
        env.close()
        runs.append(result)
        print(json.dumps({k: result[k] for k in
                          ("seed", "objectives", "actions", "learned_share")}), flush=True)

    sampled = [run for run in runs if not run["greedy"]]
    report = summarise(sampled)
    report["greedy"] = next(run for run in runs if run["greedy"])
    report["budget"] = args.budget
    report["ceiling_actions"] = 487
    report["encoder"] = args.encoder
    report["policy_checkpoint"] = args.policy
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in
                      ("seeds", "reached", "passes_done_threshold",
                       "median_learned_share")}, indent=1))


if __name__ == "__main__":
    main()
