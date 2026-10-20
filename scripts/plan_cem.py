import argparse
import json
from pathlib import Path

import torch as t
import torch.nn.functional as F

from poke.actions import ACTIONS
from poke.games import resolve
from poke.goals import (
    check_dataset,
    dataset_task,
    describe_goals,
    default_device,
    goal_indices,
    load_frame,
    start_index,
)
from poke.jepa import JEPA
from poke.tasks import get as get_task


def action_lookup(device: t.device) -> t.Tensor:
    lookup_table = t.zeros((len(ACTIONS), len(ACTIONS) - 1), device=device)
    lookup_table[1:, :] = t.eye(len(ACTIONS) - 1, device=device)
    return lookup_table

@t.inference_mode()
def plan_costs(
    model: JEPA,
    initial: t.Tensor,
    goals: t.Tensor,
    action_sequences: t.Tensor,
) -> t.Tensor:
    actions = action_lookup(initial.device)[action_sequences]
    initial_states = initial.expand(action_sequences.size(0), -1)
    predicted_states = model.rollout(initial_states, actions)
    distances = (
        predicted_states[:, :, None, :] - goals[None, None, :, :]
    ).square().mean(dim=-1)
    return distances.flatten(start_dim=1).min(dim=1).values


@t.inference_mode()
def cem_plan(
    model: JEPA,
    initial: t.Tensor,
    goals: t.Tensor,
    horizon: int = 14,
    samples: int = 512,
    elites: int = 64,
    iterations: int = 15,
    smoothing: float = 0.2,
    min_probability: float = 0.01,
    seed: int = 0,
    verbose: bool = True,
) -> tuple[t.Tensor, float, list[float]]:
    """Use categorical Cross-Entropy Method to find an action sequence."""

    if horizon < 1 or samples < 1 or iterations < 1:
        raise ValueError("horizon, samples, and iterations must be positive")
    if not 0 < elites <= samples:
        raise ValueError("elites must be between 1 and samples")
    if not 0 <= smoothing < 1:
        raise ValueError("smoothing must be in [0, 1)")
    if not 0 <= min_probability < 1 / len(ACTIONS):
        raise ValueError(
            "min_probability must be in [0, 1 / number of actions)"
        )

    device = initial.device
    generator = t.Generator(device=device).manual_seed(seed)

    probabilities = t.full(
        (horizon, len(ACTIONS)),
        1.0 / len(ACTIONS),
        device=device,
    )

    # Track the best individual sequence across every iteration.
    best_sequence = t.empty(horizon, dtype=t.long, device=device)
    best_cost = float("inf")
    cost_history: list[float] = []

    for iteration in range(iterations):
        sequences = t.multinomial(
            probabilities,
            samples,
            replacement=True,
            generator=generator,
        ).transpose(0, 1)

        costs = plan_costs(model, initial, goals, sequences)
        elite_indices = costs.topk(elites, largest=False).indices
        elite_sequences = sequences[elite_indices]

        iteration_best_index = costs.argmin()
        iteration_best_cost = costs[iteration_best_index].item()
        if iteration_best_cost < best_cost:
            best_cost = iteration_best_cost
            best_sequence = sequences[iteration_best_index].clone()

        cost_history.append(best_cost)

        elite_probabilities = F.one_hot(
            elite_sequences,
            num_classes=len(ACTIONS),
        ).float().mean(dim=0)

        probabilities = (
            smoothing * probabilities
            + (1.0 - smoothing) * elite_probabilities
        )
        probabilities = probabilities.clamp_min(min_probability)
        probabilities = probabilities / probabilities.sum(
            dim=-1,
            keepdim=True,
        )

        if verbose:
            print(
                f"cem iteration={iteration + 1:02d} "
                f"best_cost={best_cost:.6f}"
            )

    return best_sequence.cpu(), best_cost, cost_history


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search for a discrete action plan with categorical CEM"
    )
    parser.add_argument("checkpoint")
    parser.add_argument("--rom", help="optional; cross-checks the dataset game")
    parser.add_argument("--task")
    parser.add_argument("--data", default="episodes/poke.h5")
    parser.add_argument("--out", default="plans/poke.json")
    parser.add_argument("--start-index", type=int)
    parser.add_argument(
        "--goal-index",
        dest="goal_indices",
        type=int,
        action="append",
        help="goal frame index; repeat to provide multiple valid goals",
    )
    parser.add_argument("--max-goals", type=int, default=8)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--elites", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--smoothing", type=float, default=0.2)
    parser.add_argument("--min-probability", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--game")
    parser.add_argument("--device", default=default_device())
    args = parser.parse_args()

    # Planning never touches the emulator, so the ROM is optional here: the
    # dataset records the game and task it was collected for.
    data_game, data_task = dataset_task(args.data)
    game_key = resolve(args.rom, args.game).key if args.rom else data_game
    task = get_task(args.task or data_task, game_key)
    check_dataset(args.data, game_key, task.key)
    horizon = args.horizon or task.horizon

    device = t.device(args.device)
    checkpoint = t.load(args.checkpoint, map_location=device, weights_only=False)
    model = JEPA(num_frames=checkpoint["config"]["num_frames"]).to(device)
    model.load_state_dict(checkpoint["jepa"])
    # eval() matters beyond determinism: the predictor's attention carries
    # dropout=0.1, and MPS has no scaled_dot_product_attention kernel with
    # dropout, so a rollout in train mode is both stochastic and a crash here.
    model.eval()

    start = start_index(args.data) if args.start_index is None else args.start_index
    goals_at = (
        goal_indices(args.data, limit=args.max_goals)
        if args.goal_indices is None
        else args.goal_indices
    )

    print(f"goals: {describe_goals(args.data, list(goals_at))}")

    with t.inference_mode():
        initial = model.encode(load_frame(args.data, start).to(device))
        goals = t.cat(
            [model.encode(load_frame(args.data, index).to(device)) for index in goals_at]
        )

    sequence, cost, history = cem_plan(
        model,
        initial,
        goals,
        horizon=horizon,
        samples=args.samples,
        elites=args.elites,
        iterations=args.iterations,
        smoothing=args.smoothing,
        min_probability=args.min_probability,
        seed=args.seed,
        verbose=not args.quiet,
    )

    action_indices = sequence.tolist()
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "data": str(Path(args.data).resolve()),
        "game": game_key,
        "task": task.key,
        "start_index": start,
        "goal_indices": list(goals_at),
        "predicted_cost": cost,
        "cost_history": history,
        "action_indices": action_indices,
        "actions": [ACTIONS[index] for index in action_indices],
        "planner": {
            "horizon": horizon,
            "samples": args.samples,
            "elites": args.elites,
            "iterations": args.iterations,
            "smoothing": args.smoothing,
            "min_probability": args.min_probability,
            "seed": args.seed,
            "device": str(device),
        },
    }

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")

    print(f"plan: {' '.join(result['actions'])}")
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
