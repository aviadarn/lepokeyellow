"""Can the planner cross one open map? Isolated from everything else.

End-to-end runs confound navigation with dialogue, battles and waypoint
bookkeeping -- one change improved navigation and broke the text masher at the
same time, and the overall number went down. This asks the single question that
is actually blocking: starting in Pallet Town, can the planner reach the tile
that triggers Oak's encounter?

Target (0, 0, 10) is the north edge. The planner stalls around y=5, so "min y
reached" is the measurement, and reaching y=0 is success.

  uv run python -m scripts.nav_bench checkpoints/yellow_v2/best.pt
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch as t

from poke import coords_probe
from poke.env import PokeEnv
from poke.goals import default_device
from poke.jepa import JEPA
from poke.ram import coords, cur_map
from poke.walkmap import build as build_walkmap, lookup as field_cost
from scripts.plan_cem import action_lookup, cem_plan
from scripts.play import to_tensor

TARGET = (0, 0, 10)


def run(model, env, probe, lookup, device, horizon, repeat, samples,
        iterations, budget, seed, field=None, how="discounted",
        discount=0.9, costs=None):
    """One MPC run. `costs` is the dense cost field as numpy, for reporting:
    `min y` only says how far north the player got, and the route turns east
    before it turns north, so distance to the target is the honest number."""
    env.reset()
    maps = probe["maps"].tolist()
    map_index = maps.index(TARGET[0])
    index_of = {map_id: index for index, map_id in enumerate(maps)}
    best_y, best_cost, spent = 99, float("inf"), 0
    trace = []
    while spent < budget:
        here = (cur_map(env), *coords(env))
        if costs is not None:
            best_cost = min(best_cost, field_cost(
                costs, index_of.get(int(here[0])), here[1], here[2],
                float(costs.max()),
            ))
        if here[0] == TARGET[0] and here[1] is not None:
            best_y = min(best_y, here[1])
            if here[1] == TARGET[1]:
                return {"reached": True, "actions": spent, "min_y": 0,
                        "min_cost": 0.0, "trace": trace}
        with t.inference_mode():
            initial = model.encode(to_tensor(env.screen(), device))
        sequence, _, _ = cem_plan(
            model, initial, None, horizon=horizon, samples=samples,
            elites=max(8, samples // 8), iterations=iterations,
            seed=seed + spent, verbose=False,
            score_fn=lambda m, i, s: coords_probe.plan_costs(
                m, i, s, probe, lookup, map_index, TARGET[1], TARGET[2],
                repeat=repeat, field=field, how=how, discount=discount,
            ),
        )
        for action_index in sequence.tolist()[:2]:
            for _ in range(repeat):
                env.step(action_index)
                spent += 1
        trace.append((cur_map(env), *coords(env)))
    return {"reached": False, "actions": spent, "min_y": best_y,
            "min_cost": best_cost, "trace": trace[-5:]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("--rom", default="pokeyellow.gbc")
    parser.add_argument("--state", default="states/pallet.state")
    parser.add_argument("--data", default="episodes/yellow_v3.h5")
    parser.add_argument("--horizons", type=int, nargs="+", default=[14, 30])
    parser.add_argument("--repeats", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--samples", type=int, default=384)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--budget", type=int, default=150)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--walkmap", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="score on BFS hops over observed transitions "
                             "instead of Manhattan distance")
    parser.add_argument("--fit-on", default="real",
                        choices=("real", "predicted", "both"),
                        help="fit the coordinate head on real encodings, on "
                             "predicted embeddings (what the planner decodes), "
                             "or on both")
    parser.add_argument("--fit-steps", type=int, default=2,
                        help="rollout length used to build predicted features")
    parser.add_argument("--unknown-cost", type=float, default=60.0)
    parser.add_argument("--head", default="mlp", choices=("linear", "mlp"),
                        help="coordinate read-out head to plan on")
    parser.add_argument("--how", nargs="+", default=["discounted"],
                        choices=("min", "final", "discounted"),
                        help="how a plan's cost trace collapses to one number")
    parser.add_argument("--discount", type=float, default=0.9)
    parser.add_argument("--out", default="results/nav_bench.json")
    parser.add_argument("--device", default=default_device())
    args = parser.parse_args()

    device = t.device(args.device)
    checkpoint = t.load(args.checkpoint, map_location=device, weights_only=False)
    model = JEPA(num_frames=checkpoint["config"]["num_frames"]).to(device)
    model.load_state_dict(checkpoint["jepa"])
    model.eval()
    fitted = coords_probe.fit(model, args.data, device, head=args.head,
                                 on=args.fit_on, steps=args.fit_steps)
    probe = fitted.to_torch(device)
    lookup = action_lookup(device)
    print(f"{fitted.head} position head: +/-{fitted.tile_mae:.2f} tiles "
          f"(linear in-sample {fitted.linear_tile_mae:.2f}), map id "
          f"{fitted.map_accuracy:.1%} over {len(fitted.maps)} maps; "
          f"frames per map {fitted.frames_per_map}")

    field = grids = None
    if args.walkmap:
        walk = build_walkmap(args.data)
        grids = walk.fields(fitted.maps, TARGET, (32, 32),
                            unknown=args.unknown_cost)
        field = t.from_numpy(grids).to(device)
        reachable = len(walk.global_distances(TARGET))
        print(f"walkmap: {len(walk.nodes())} tiles across {len(fitted.maps)} "
              f"maps, {reachable} can reach the target")

    env = PokeEnv(args.rom, args.state, headless=True)
    env.reset()
    start = (cur_map(env), *coords(env))
    print(f"start {start} -> target {TARGET}; budget {args.budget} actions, "
          f"{args.seeds} seeds\n")
    print(f"{'horizon':>8}{'repeat':>8}{'how':>12}{'lookahead':>11}"
          f"{'reached':>9}{'best y':>8}{'median y':>10}{'best':>9}{'med cost':>11}")

    rows = []
    try:
        for horizon, repeat, how in itertools.product(
            args.horizons, args.repeats, args.how
        ):
            results = [
                run(model, env, probe, lookup, device, horizon, repeat,
                    args.samples, args.iterations, args.budget, seed * 1000,
                    field, how, args.discount, grids)
                for seed in range(args.seeds)
            ]
            ys = [r["min_y"] for r in results]
            cs = [r["min_cost"] for r in results]
            hits = sum(r["reached"] for r in results)
            rows.append({"horizon": horizon, "repeat": repeat, "how": how,
                         "lookahead": horizon * repeat,
                         "reached": hits, "seeds": args.seeds,
                         "best_y": int(min(ys)), "median_y": float(np.median(ys)),
                         "best_cost": float(min(cs)),
                         "median_cost": float(np.median(cs))})
            print(f"{horizon:>8}{repeat:>8}{how:>12}{horizon * repeat:>11}"
                  f"{f'{hits}/{args.seeds}':>9}{min(ys):>8}{np.median(ys):>10.1f}"
                  f"{min(cs):>9.0f}{np.median(cs):>11.1f}")
    finally:
        env.close()

    best = min(rows, key=lambda r: (-r["reached"], r["median_cost"]))
    print(f"\nbest: horizon {best['horizon']} x repeat {best['repeat']} "
          f"scored by {best['how']} = {best['lookahead']} actions of lookahead, "
          f"reached {best['reached']}/{best['seeds']}, median cost "
          f"{best['median_cost']} from {rows and args.budget and ''}"
          f"{best['median_y']} min y")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"start": list(start), "target": list(TARGET), "budget": args.budget,
         "rows": rows}, indent=2) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
