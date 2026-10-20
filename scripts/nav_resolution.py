"""Per action, what tile does the model think it lands on, and the real one?

This is the measurement that diagnosed the navigation stall, so it is a script
rather than something retyped each time. For one state it prints, for all seven
buttons, the tile the coordinate probe reads off the PREDICTED embedding beside
the tile the emulator actually produces, and the cost field's value at each.

What it showed with the linear head: predicted x sat in 6.8-7.5 for every
button while the truth was 5. `right` genuinely improves the BFS distance from
6 to 5 and the model scored it 7, identical to standing still. Adjacent levels
of the field differ by one hop and the head's error was 1.44-1.88 tiles, so the
planner was choosing between options it could not tell apart.

Each real action is taken from a snapshot and rolled back, so the sequence of
probes does not disturb the state being measured.

  uv run python -m scripts.nav_resolution checkpoints/yellow_v2/best.pt
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
    parser.add_argument("--prefix", nargs="*", default=["down"],
                        help="actions to take from the save state before "
                             "measuring, so the start tile can be chosen")
    parser.add_argument("--fit-on", default="real",
                        choices=("real", "predicted", "both"),
                        help="fit the coordinate head on real encodings, on "
                             "predicted embeddings (what the planner decodes), "
                             "or on both")
    parser.add_argument("--fit-steps", type=int, default=2,
                        help="rollout length used to build predicted features")
    parser.add_argument("--unknown-cost", type=float, default=60.0)
    parser.add_argument("--out", default="results/nav_resolution.json")
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
    print(f"{fitted.head} head: +/-{fitted.tile_mae:.2f} tiles, map id "
          f"{fitted.map_accuracy:.1%}")
    if fitted.holdout_tile_mae:
        print("  held out per map: " + "  ".join(
            f"{k}={v:.2f}" for k, v in sorted(fitted.holdout_tile_mae.items())))

    walk = build_walkmap(args.data)
    field = walk.fields(fitted.maps, TARGET, (32, 32), unknown=args.unknown_cost)
    target_map_index = fitted.maps.index(TARGET[0])
    index_of = {map_id: index for index, map_id in enumerate(fitted.maps)}
    field_at = lambda map_id, y, x: field_cost(field, index_of.get(int(map_id)), y, x,
                                           args.unknown_cost)

    env = PokeEnv(args.rom, args.state, headless=True)
    try:
        env.reset()
        for name in args.prefix:
            env.step(ACTIONS.index(name))
        here = (cur_map(env), *coords(env))
        start_cost = field_at(here[0], here[1], here[2])
        print(f"start {here}, {start_cost:.0f} hops from {TARGET}\n")
        print(f"{'action':>7}{'rollout+head':>16}{'cost':>6}"
              f"{'head on real':>16}{'cost':>6}{'real tile':>13}{'cost':>6}")

        rows = []
        for index, name in enumerate(ACTIONS):
            blob = env.snapshot()
            with t.inference_mode():
                initial = model.encode(to_tensor(env.screen(), device))
                sequence = t.tensor([[index]], device=device)
                states = model.rollout(initial, lookup[sequence])
                map_index, y, x = coords_probe.decode(states, probe)
            predicted = (int(map_index[0, 0]), float(y[0, 0]), float(x[0, 0]))
            on_target_map = predicted[0] == target_map_index
            predicted_cost = field_at(fitted.maps[predicted[0]],
                                      predicted[1], predicted[2])

            env.step(index)
            real = (cur_map(env), *coords(env))
            # The head applied to the frame the emulator ACTUALLY produced.
            # Between this column and the rollout column sits every error the
            # predictor makes; between this column and the real tile sits every
            # error the head makes. Without the split, one number carries both.
            with t.inference_mode():
                observed = model.encode(to_tensor(env.screen(), device))
                seen_map, seen_y, seen_x = coords_probe.decode(
                    observed[:, None, :], probe
                )
            env.restore(blob)

            seen = (int(seen_map[0, 0]), float(seen_y[0, 0]), float(seen_x[0, 0]))
            seen_on_target = seen[0] == target_map_index
            seen_cost = field_at(fitted.maps[seen[0]], seen[1], seen[2])
            real_cost = field_at(real[0], real[1], real[2])

            rows.append({
                "action": name,
                "predicted_map": fitted.maps[predicted[0]],
                "predicted": [round(predicted[1], 2), round(predicted[2], 2)],
                "predicted_cost": predicted_cost,
                "observed_map": fitted.maps[seen[0]],
                "observed": [round(seen[1], 2), round(seen[2], 2)],
                "observed_cost": seen_cost,
                "real_map": real[0],
                "real": [real[1], real[2]],
                "real_cost": real_cost,
            })
            tile = lambda on_map, where, map_id: (
                f"({where[0]:.1f}, {where[1]:.1f})" if on_map else f"map {map_id}"
            )
            print(f"{name:>7}"
                  f"{tile(on_target_map, predicted[1:], fitted.maps[predicted[0]]):>16}"
                  f"{predicted_cost:>6.0f}"
                  f"{tile(seen_on_target, seen[1:], fitted.maps[seen[0]]):>16}"
                  f"{seen_cost:>6.0f}"
                  f"{(f'({real[1]}, {real[2]})' if real[0] == TARGET[0] else f'map {real[0]}'):>13}"
                  f"{real_cost:>6.0f}")
    finally:
        env.close()

    # The planner commits the action it scores best. Does that action actually
    # reduce the true distance? That is the whole question, in one line.
    best = min(rows, key=lambda r: r["predicted_cost"])
    truly_best = min(rows, key=lambda r: r["real_cost"])
    agrees = best["real_cost"] <= truly_best["real_cost"] + 1e-9
    improves = best["real_cost"] < start_cost
    print(f"\nchosen {best['action']!r}: real cost {best['real_cost']:.0f} "
          f"from {start_cost:.0f}; best available {truly_best['action']!r} at "
          f"{truly_best['real_cost']:.0f}")
    print("the head picks a tile-reducing action"
          if improves else
          "the head picks an action that does NOT reduce the true distance")

    # Same question again, but scoring on the head applied to the real next
    # screen. If that choice is right and the rollout one is wrong, the
    # predictor is the problem; if both are wrong, the head still is.
    oracle = min(rows, key=lambda r: r["observed_cost"])
    print(f"with a perfect predictor it would choose {oracle['action']!r}: "
          f"real cost {oracle['real_cost']:.0f}")
    on_map = [r for r in rows if r["observed_map"] == TARGET[0]
              and r["real_map"] == TARGET[0]]
    head_error = float(np.mean([
        abs(r["observed"][0] - r["real"][0]) + abs(r["observed"][1] - r["real"][1])
        for r in on_map
    ])) if on_map else float("nan")
    rollout_error = float(np.mean([
        abs(r["predicted"][0] - r["observed"][0])
        + abs(r["predicted"][1] - r["observed"][1])
        for r in rows if r["predicted_map"] == TARGET[0]
        and r["observed_map"] == TARGET[0]
    ])) if on_map else float("nan")
    print(f"head error on real screens {head_error:.2f} tiles; rollout adds "
          f"{rollout_error:.2f}")

    spread = [r["predicted"][1] for r in rows if r["predicted_map"] == TARGET[0]]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "head": fitted.head,
        "tile_mae": fitted.tile_mae,
        "start": list(here),
        "target": list(TARGET),
        "start_cost": start_cost,
        "rows": rows,
        "chosen": best["action"],
        "chosen_real_cost": best["real_cost"],
        "best_available": truly_best["action"],
        "best_real_cost": truly_best["real_cost"],
        "agrees_with_best": bool(agrees),
        "reduces_distance": bool(improves),
        "predicted_x_spread": [min(spread), max(spread)] if spread else None,
        "oracle_action": oracle["action"],
        "oracle_real_cost": oracle["real_cost"],
        "head_error_tiles": head_error,
        "rollout_error_tiles": rollout_error,
        "holdout_tile_mae": fitted.holdout_tile_mae,
    }, indent=2) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
