"""DAgger for Part 2: relabel the states the clone actually reaches.

TRAINING ONLY. The oracle snapshots and restores, which the play-time contract
bans; the rollout it is given states from runs under the contract, through the
same `scripts/play2.play` loop the baselines and the eval use.

Why. Behaviour cloning on the oracle's solution fit to loss 0.0005 and scored
0 of 11 under the contract: the clone walked the route correctly for five
decisions, then oscillated between two tiles for four hundred actions
(`results/part2_clone_eval.json`). The dataset is one optimal trajectory per
objective and holds no frame showing the way back from a mistake, because the
oracle never made one. So the clone's sixth decision is taken on a screen its
training set does not contain, and every decision after that is further out.

The loop, per round:

  1. roll the current clone out under the contract, keeping a save state
     behind each new decision (`poke.rl.dagger.SnapshotProbe`);
  2. pick a handful of those states, spread over the rollout;
  3. restore each and run the oracle from it -- the expert, queried on the
     learner's own distribution, which is the thing one optimal trajectory
     cannot supply;
  4. append the solutions, warmed with the learner's own preceding turns, to
     the dataset;
  5. retrain the clone on everything collected so far.

Each round's new demonstrations go to their own file, so a round can be
dropped by deleting one file and the training set is the list of files.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from poke.actions import ACTIONS, to_multihot
from poke.contract import Run
from poke.env import PokeEnv
from poke.goals import default_device
from poke.objectives import CHAIN
from poke.oracle import nodes_for, search
from poke.ram import coords, cur_map, party_count
from poke.rl.clone import save_clone, train_clone
from poke.rl.dagger import SnapshotProbe, by_time_spent, spread
from poke.rl.features import FrozenEncoder
from poke.rl.policy import ActorCritic
from poke.writer import EpisodeWriter
from scripts.collect_demos import PHASE_OBJECTIVE, demo_steps, policy_turns
from scripts.eval_part2 import BUDGET, NetPolicy, write_gif
from scripts.play2 import play
from scripts.train_clone import load_episodes

#: The oracle phase that solves each Part 2 objective.
OBJECTIVE_PHASE = {objective: phase for phase, objective in PHASE_OBJECTIVE.items()}


def goal_for(env, objective_key: str):
    """(goal predicate, allowed maps, buttons) for the objective in progress.

    `scripts/oracle.phases` lists the maps each phase's route crossed starting
    from that phase's own start. A clone that wandered can be standing off
    that list, and `search` prunes every child whose map is not allowed -- so
    the map the player is on now is added, or the search is boxed in at its
    own root.
    """
    from scripts.oracle import phases

    name = OBJECTIVE_PHASE.get(objective_key)
    if name is None:
        raise ValueError(
            f"{objective_key!r} has no oracle phase. Part 2 is "
            + ", ".join(sorted(OBJECTIVE_PHASE))
            + "; the earlier objectives are already flagged in the start state."
        )
    for phase_name, goal, allowed, buttons in phases(env):
        if phase_name == name:
            here = cur_map(env)
            return goal, set(allowed) | ({here} if here is not None else set()), buttons
    raise ValueError(f"scripts.oracle.phases has no phase {name!r}")


class ContextStep:
    """A learner turn, in the shape `write_round` writes demonstrations in.

    It carries the action the LEARNER took, not the expert's -- the point is
    to replay the sequence that got it there. `episode_meta.goal_step` on the
    written episode records how many of these it starts with, so training
    scores nothing before them.
    """

    def __init__(self, frame, action_index):
        self.frame = frame
        self.action_name = ACTIONS[action_index]
        self.location = None
        self.policy_turn = True


def relabel(env, visit, max_nodes: int | None = None):
    """What the expert would have done from a state the learner reached.

    Returns (demonstration steps, the search result). The steps come back the
    way `scripts/collect_demos.py` builds them -- one per button actually
    pressed, each paired with the frame it was chosen from, each flagged with
    whether a contract-bound policy would have owned the turn.
    """
    env.restore(visit.blob)
    goal, allowed, buttons = goal_for(env, visit.objective)
    result = search(env, goal, allowed, buttons=buttons,
                    with_facing="a" in buttons,
                    max_nodes=max_nodes or nodes_for())
    if not result.found:
        return [], result
    return demo_steps(env, result.edge_blobs, result.edge_traces), result


def rollout(env, policy_path, encoder, device, budget, context, greedy=True, seed=0):
    """One contract-bound episode, with the state behind each new decision kept."""
    net = ActorCritic(feature_dim=encoder.dim, num_frames=encoder.num_frames).to(device)
    blob = torch.load(policy_path, map_location="cpu", weights_only=False)
    net.load_state_dict(blob["policy"])
    net.eval()
    inner = NetPolicy(net, encoder, device=device, greedy=greedy, seed=seed)
    probe = SnapshotProbe(inner, env, context=context)
    result = play(env, probe, budget=budget, run=Run(env))
    return probe, inner, result


def write_round(path, rom, state, episodes) -> int:
    """One h5 per round: [(steps, objective key, context length)] -> file."""
    written = 0
    with EpisodeWriter(path, rom, state, 60, 0, "yellow", task="dagger_part2") as writer:
        for steps, key, context in episodes:
            for step in steps:
                writer.add_step(step.frame, to_multihot(ACTIONS.index(step.action_name)),
                                0, False, step.location)
            writer.end_episode("dagger", key, True, -1, context)
            written += len(steps)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rom", default="pokeyellow.gbc")
    parser.add_argument("--state", default="states/oracle/route1.state")
    parser.add_argument("--encoder", default="checkpoints/part2/best.pt")
    parser.add_argument("--clone", default="checkpoints/rl/clone.pt",
                        help="the clone to start from; rewritten each round")
    parser.add_argument("--demos", nargs="+", default=["episodes/demos_part2.h5"],
                        help="the dataset so far; each round appends a file")
    parser.add_argument("--episodes-dir", default="episodes")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--start-round", type=int, default=1,
                        help="numbering for this run's round files, so a "
                             "continuation does not overwrite earlier ones")
    parser.add_argument("--pick", default="stuck", choices=("stuck", "spread"),
                        help="which visited states to relabel. `stuck` takes "
                             "the ones the rollout spent the most turns in; "
                             "`spread` takes them evenly over the rollout, "
                             "which measurement showed spends the budget on "
                             "the prefix the clone already walks correctly")
    parser.add_argument("--states", type=int, default=6,
                        help="states relabelled per round. One oracle search "
                             "costs about five minutes.")
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--context", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-nodes", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gif-dir", default="media")
    parser.add_argument("--out", default="results/dagger_part2.json")
    args = parser.parse_args()

    encoder = FrozenEncoder(args.encoder, device=args.device, half=args.half)
    demo_files = list(args.demos)
    report = {"state": args.state, "encoder": args.encoder, "rounds": [],
              "states_per_round": args.states, "pick": args.pick,
              "budget": args.budget,
              "context": args.context,
              "what": "DAgger: the oracle relabels the states the clone reaches.",
              "contract": "training only: the relabelling search uses snapshots"}

    for index in range(args.start_round, args.start_round + args.rounds):
        print(f"=== round {index} ===", flush=True)
        env = PokeEnv(args.rom, args.state)
        env.reset()
        # No settle(): it shifts the frame phase, measured in Stage 1.
        probe, net_policy, played = rollout(
            env, args.clone, encoder, args.device, args.budget, args.context,
            seed=args.seed)
        gif = Path(args.gif_dir) / f"dagger_r{index}.gif"
        write_gif(net_policy.frames, gif)
        print(json.dumps({k: played[k] for k in
                          ("objectives", "actions", "learned_share")}), flush=True)
        print(f"{probe.turns} policy turns, {len(probe.visits)} distinct states, "
              f"{probe.revisits} revisits", flush=True)

        pick = by_time_spent if args.pick == "stuck" else spread
        chosen = pick(probe.visits, args.states)
        relabelled, episodes = [], []
        for visit in chosen:
            steps, result = relabel(env, visit, max_nodes=args.max_nodes)
            kept = policy_turns(steps)
            context = visit.context if kept else []
            if kept:
                # The learner's own turns first, unscored, then the expert's.
                episodes.append((
                    [ContextStep(frame, action) for frame, action in context] + kept,
                    visit.objective, len(context)))
            relabelled.append({
                "turn": visit.turn, "revisits": visit.revisits,
                "objective": visit.objective,
                "found": result.found, "nodes": result.nodes,
                "seconds": result.seconds, "steps": len(kept),
                "steps_before_filter": len(steps), "context": len(context),
                "end": result.end,
            })
            print(json.dumps(relabelled[-1]), flush=True)
        env.close()

        path = Path(args.episodes_dir) / f"dagger_r{index}.h5"
        written = write_round(path, args.rom, args.state, episodes)
        demo_files.append(str(path))

        loaded = []
        for file in demo_files:
            loaded.extend(load_episodes(file))
        # Warm-up turns are replayed and not scored, so they are not demos.
        scored = sum(len(actions) - context for _, actions, _, context in loaded)
        print(f"retraining on {scored} scored steps from {len(demo_files)} files",
              flush=True)
        policy, history = train_clone(encoder, loaded, epochs=args.epochs,
                                      lr=args.lr, seed=args.seed,
                                      log_every=max(args.epochs // 6, 1))
        Path(args.clone).parent.mkdir(parents=True, exist_ok=True)
        save_clone(args.clone, policy, encoder.dim, encoder.num_frames,
                   demos=scored, history=history)

        report["rounds"].append({
            "round": index,
            "rollout": {k: played[k] for k in
                        ("objectives", "actions", "by_controller", "learned_share")},
            "policy_turns": probe.turns,
            "distinct_states": len(probe.visits),
            "revisits": probe.revisits,
            "relabelled": relabelled,
            "new_steps": written,
            "dataset_files": list(demo_files),
            "dataset_steps": scored,
            "first_loss": round(history[0], 4),
            "final_loss": round(history[-1], 4),
            "gif": str(gif),
        })
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1))

    print(json.dumps({"rounds": len(report["rounds"]),
                      "reached": [r["rollout"]["objectives"] for r in report["rounds"]]},
                     indent=1))


if __name__ == "__main__":
    main()
