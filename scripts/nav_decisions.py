"""How often does the cost the planner computes pick an action that helps?

`nav_bench.py` reports whether a full CEM run reached the target, which is a
pass/fail on a long chain of things. `nav_resolution.py` opens up one state.
This is the quantity in between, and the one to judge a navigation change on:
over many states, how often does the action with the lowest PREDICTED cost
actually reduce the true distance to the target?

At each state every one of the seven buttons is evaluated twice -- once by
rolling the predictor forward one step and decoding into the cost field, once
by taking it in the emulator and rolling back -- and then the walk advances by
the action the predicted cost chose. So the states measured are the states this
cost function would actually visit, including the ones it walks itself into.

Three numbers come out. `optimal` is how often the chosen action was among the
best available, which is strict and will never be 1.0 because several actions
often tie. `helps` is how often it strictly reduced the true distance, which is
all a planner needs to make progress. `no worse` is the tolerant version, and
the gap between it and `helps` is how much of the walk is standing still.

  uv run python -m scripts.nav_decisions checkpoints/yellow_v2/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch as t

from poke import coords_probe
from poke.actions import ACTIONS
from poke.env import PokeEnv
from poke.goals import default_device
from poke.jepa import JEPA
from poke.ram import coords, cur_map
from poke.walkmap import build as build_walkmap, lookup as field_cost
from scripts.nav_bench import TARGET
from scripts.plan_cem import action_lookup
from scripts.play import to_tensor


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("checkpoint")
    parser.add_argument("--rom", default="pokeyellow.gbc")
    parser.add_argument("--state", default="states/pallet.state")
    parser.add_argument("--data", default="episodes/yellow_v3.h5")
    parser.add_argument("--head", default="mlp", choices=("linear", "mlp"))
    parser.add_argument("--fit-on", default="real",
                        choices=("real", "predicted", "both"))
    parser.add_argument("--fit-steps", type=int, default=2)
    parser.add_argument("--steps", type=int, default=60,
                        help="decisions to measure")
    parser.add_argument("--unknown-cost", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/nav_decisions.json")
    parser.add_argument("--device", default=default_device())
    args = parser.parse_args()

    device = t.device(args.device)
    checkpoint = t.load(args.checkpoint, map_location=device, weights_only=False)
    model = JEPA(num_frames=checkpoint["config"]["num_frames"]).to(device)
    model.load_state_dict(checkpoint["jepa"])
    model.eval()

    fitted = coords_probe.fit(model, args.data, device, head=args.head,
                              on=args.fit_on, steps=args.fit_steps,
                              seed=args.seed)
    probe = fitted.to_torch(device)
    lookup = action_lookup(device)
    walk = build_walkmap(args.data)
    field = walk.fields(fitted.maps, TARGET, (32, 32), unknown=args.unknown_cost)
    index_of = {map_id: index for index, map_id in enumerate(fitted.maps)}
    cost_of = lambda map_id, y, x: field_cost(field, index_of.get(int(map_id)), y, x,
                                          args.unknown_cost)
    print(f"{fitted.head} head on {fitted.fitted_on} embeddings, "
          f"+/-{fitted.tile_mae:.2f} tiles held out")

    env = PokeEnv(args.rom, args.state, headless=True)
    decisions = []
    try:
        env.reset()
        env.settle()
        for step in range(args.steps):
            here = (cur_map(env), *coords(env))
            start_cost = cost_of(*here)
            blob = env.snapshot()

            predicted_costs, real_costs = [], []
            with t.inference_mode():
                initial = model.encode(to_tensor(env.screen(), device))
                sequence = t.arange(len(ACTIONS), device=device)[:, None]
                states = model.rollout(
                    initial.expand(len(ACTIONS), -1), lookup[sequence]
                )
                map_index, y, x = coords_probe.decode(states, probe)
            for index in range(len(ACTIONS)):
                predicted_costs.append(cost_of(
                    fitted.maps[int(map_index[index, 0])],
                    float(y[index, 0]), float(x[index, 0]),
                ))
                env.step(index)
                real_costs.append(cost_of(cur_map(env), *coords(env)))
                env.restore(blob)

            chosen = int(np.argmin(predicted_costs))
            best = float(np.min(real_costs))
            decisions.append({
                "step": step,
                "at": list(here),
                "start_cost": start_cost,
                "chosen": ACTIONS[chosen],
                "chosen_real_cost": real_costs[chosen],
                "best_real_cost": best,
                "optimal": real_costs[chosen] <= best + 1e-9,
                "helps": real_costs[chosen] < start_cost,
                "no_worse": real_costs[chosen] <= start_cost,
            })
            env.step(chosen)
            if cur_map(env) == TARGET[0] and coords(env)[0] == TARGET[1]:
                break
    finally:
        env.close()

    rate = lambda key: float(np.mean([d[key] for d in decisions]))
    costs = [d["start_cost"] for d in decisions] + [decisions[-1]["chosen_real_cost"]]
    summary = {
        "steps": len(decisions),
        "optimal": rate("optimal"),
        "helps": rate("helps"),
        "no_worse": rate("no_worse"),
        "start_cost": costs[0],
        "final_cost": costs[-1],
        "best_cost_seen": float(min(costs)),
        "reached": bool(min(costs) == 0),
    }
    print(f"\n{summary['steps']} decisions from cost {summary['start_cost']:.0f}")
    print(f"  optimal  {summary['optimal']:.1%}")
    print(f"  helps    {summary['helps']:.1%}")
    print(f"  no worse {summary['no_worse']:.1%}")
    print(f"  cost {summary['start_cost']:.0f} -> {summary['final_cost']:.0f} "
          f"(best seen {summary['best_cost_seen']:.0f})")
    visited = {tuple(d["at"]) for d in decisions}
    print(f"  {len(visited)} distinct tiles visited in {summary['steps']} steps")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "head": fitted.head,
        "fitted_on": fitted.fitted_on,
        "tile_mae": fitted.tile_mae,
        "holdout_tile_mae": fitted.holdout_tile_mae,
        "target": list(TARGET),
        "summary": summary,
        "decisions": decisions,
    }, indent=2) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
