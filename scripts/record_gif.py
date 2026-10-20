"""Record what the player actually does, as an animated GIF.

Every other script in this repo reports navigation as a number. This writes the
frames, with the state of the world printed under them, so the failure is
visible rather than inferred: the planner walking into the player's own house,
or oscillating between two tiles, is obvious in two seconds of animation and
takes a table to argue for.

Four policies, all from the same save state so they are comparable:

  cem       the real planner -- CEM over button sequences scored by the
            coordinate probe against the walkmap cost field, replanning every
            `--commit` actions
  greedy    one-step argmin over the seven buttons, the same cost without the
            search, which isolates the cost signal from the planner
  route     the recorded human demonstration, as the reference for what the
            task looks like done
  random    uniform buttons, the baseline every planner number is measured
            against

The caption carries the emulator's truth (map, tile, hops to target) and, for
the two model policies, the tile the model predicted -- so a frame where they
disagree is a frame where the planner was misinformed.

  uv run python -m scripts.record_gif checkpoints/yellow_v2/best.pt \
      --policy cem --out media/planner.gif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch as t
from PIL import Image, ImageDraw, ImageFont

from poke import coords_probe
from poke.actions import ACTIONS
from poke.env import PokeEnv
from poke.goals import default_device
from poke.jepa import JEPA
from poke.maps import name as map_name
from poke.ram import coords, cur_map
from poke.routes import Route, state_digest
from poke.walkmap import build as build_walkmap, lookup as field_cost
from scripts.nav_bench import TARGET
from scripts.plan_cem import action_lookup, cem_plan
from scripts.play import to_tensor

CAPTION_HEIGHT = 40


def caption_font(scale: int) -> ImageFont.ImageFont:
    """Readable at whatever scale the screen is drawn at."""
    try:
        return ImageFont.load_default(size=8 + 4 * scale)
    except TypeError:            # Pillow before load_default took a size
        return ImageFont.load_default()


def map_panel(field: np.ndarray, here, trail, target, size: int,
              font=None) -> Image.Image:
    """The cost field of the map the player is on, with where they have been.

    Without this the GIF hides its own subject. The camera is centred on the
    player, so walking shows up only as the background scrolling; a good part
    of any route is dialogue, where nothing moves at all; and when the planner
    is stuck it is stuck on two tiles, which looks exactly like a still image.
    Drawing the tile, the trail and the distance makes a run that is going
    nowhere look like a run that is going nowhere, instead of looking broken.

    White cells are ground never walked on in the dataset, so the planner has
    no distance for them. Every other cell is shaded by how far it is from the
    target, dark being closer. Nothing observed is ever drawn white, which is
    what lets the markers -- the ring on the target, the dot on each tile
    already visited, the square on the player -- all be white and still read.
    """
    rows, cols = field.shape
    known = field[field < field.max()]
    worst = float(known.max()) if known.size else 1.0

    cell = max(4, size // max(rows, cols))
    panel = Image.new("L", (cols * cell, rows * cell + 28), 255)
    draw = ImageDraw.Draw(panel)
    for y in range(rows):
        for x in range(cols):
            cost = float(field[y, x])
            if cost >= field.max():
                shade = 255                     # never walked on
            else:
                shade = int(40 + 150 * min(cost / worst, 1.0))
            draw.rectangle([x * cell, y * cell,
                            (x + 1) * cell - 1, (y + 1) * cell - 1], fill=shade)

    for (y, x) in trail:
        draw.rectangle([x * cell + cell // 3, y * cell + cell // 3,
                        x * cell + cell - cell // 3,
                        y * cell + cell - cell // 3], fill=255)
    ty, tx = target
    if 0 <= ty < rows and 0 <= tx < cols:
        draw.ellipse([tx * cell, ty * cell,
                      (tx + 1) * cell - 1, (ty + 1) * cell - 1],
                     outline=255, width=max(1, cell // 5))
    if here[1] is not None and 0 <= here[1] < rows and 0 <= here[2] < cols:
        draw.rectangle([here[2] * cell, here[1] * cell,
                        (here[2] + 1) * cell - 1, (here[1] + 1) * cell - 1],
                       fill=0, outline=255, width=max(2, cell // 4))
    legend = ImageFont.load_default(size=11)
    draw.text((2, rows * cell + 2), "dark = closer to the goal",
              fill=0, font=legend)
    draw.text((2, rows * cell + 14), "white = never walked on",
              fill=0, font=legend)
    return panel


def crop_field(grid: np.ndarray, unknown: float) -> tuple[np.ndarray, int, int]:
    """Trim a cost grid to the ground that was actually observed, plus a cell.

    The grids are 32x32 because that covers any Gen-1 map; Pallet Town uses 13
    rows of it. Drawing the empty remainder would shrink the part worth seeing.
    """
    known = np.argwhere(grid < unknown)
    if not len(known):
        return grid, 0, 0
    (y0, x0), (y1, x1) = known.min(axis=0), known.max(axis=0)
    y0, x0 = max(int(y0) - 1, 0), max(int(x0) - 1, 0)
    y1 = min(int(y1) + 2, grid.shape[0])
    x1 = min(int(x1) + 2, grid.shape[1])
    return grid[y0:y1, x0:x1], y0, x0


def frame_image(screen: np.ndarray, scale: int, lines: list[str],
                font=None, panel: Image.Image | None = None) -> Image.Image:
    """One GIF frame: the Game Boy screen, scaled, over a caption strip."""
    grey = Image.fromarray(screen[:, :, 0].astype(np.uint8), mode="L")
    grey = grey.resize((grey.width * scale, grey.height * scale), Image.NEAREST)

    font = font or caption_font(scale)
    step = CAPTION_HEIGHT // 2
    width = grey.width + (0 if panel is None else panel.width + 8)
    canvas = Image.new("L", (width, grey.height + CAPTION_HEIGHT), 255)
    canvas.paste(grey, (0, 0))
    if panel is not None:
        canvas.paste(panel, (grey.width + 8, 0))
    draw = ImageDraw.Draw(canvas)
    for index, text in enumerate(lines[:2]):
        draw.text((8, grey.height + 3 + index * step), text, fill=0, font=font)
    return canvas


def caption(step: int, here, cost, action: str | None, predicted=None) -> list[str]:
    where = (f"{map_name(here[0])[:18]} ({here[1]}, {here[2]})"
             if here[1] is not None else map_name(here[0])[:18])
    first = f"{step:>3}  {where}   {cost:.0f} hops to go"
    second = "" if action is None else f"     pressed {action}"
    if predicted is not None:
        second += f"   model predicted ({predicted[0]:.1f}, {predicted[1]:.1f})"
    return [first, second]


def model_step(model, env, probe, lookup, device, field, index_of, policy,
               horizon, samples, iterations, commit, seed, spent,
               field_tensor=None):
    """Choose actions the way the named policy does, and say what it predicted.

    Returns (actions to take, the tile the model predicted for the first one).
    """
    with t.inference_mode():
        initial = model.encode(to_tensor(env.screen(), device))
    if policy == "greedy":
        with t.inference_mode():
            sequence = t.arange(len(ACTIONS), device=device)[:, None]
            states = model.rollout(initial.expand(len(ACTIONS), -1),
                                   lookup[sequence])
            map_index, y, x = coords_probe.decode(states, probe)
        costs = [
            field_cost(field, index_of.get(int(probe["maps"][map_index[i, 0]])),
                       float(y[i, 0]), float(x[i, 0]), float(field.max()))
            for i in range(len(ACTIONS))
        ]
        choice = int(np.argmin(costs))
        return [choice], (float(y[choice, 0]), float(x[choice, 0]))

    target_map_index = index_of[TARGET[0]]
    sequence, _, _ = cem_plan(
        model, initial, None, horizon=horizon, samples=samples,
        elites=max(8, samples // 8), iterations=iterations,
        seed=seed + spent, verbose=False,
        score_fn=lambda m, i, s: coords_probe.plan_costs(
            m, i, s, probe, lookup, target_map_index, TARGET[1], TARGET[2],
            field=field_tensor,
        ),
    )
    chosen = sequence.tolist()[:commit]
    with t.inference_mode():
        states = model.rollout(initial, lookup[t.tensor([chosen], device=device)])
        _, y, x = coords_probe.decode(states, probe)
    return chosen, (float(y[0, -1]), float(x[0, -1]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("checkpoint", nargs="?",
                        help="not needed for --policy route or random")
    parser.add_argument("--rom", default="pokeyellow.gbc")
    parser.add_argument("--state", default="states/pallet.state")
    parser.add_argument("--data", default="episodes/yellow_v3.h5")
    parser.add_argument("--policy", default="cem",
                        choices=("cem", "greedy", "route", "random"))
    parser.add_argument("--route", default="routes/yellow_pikachu.json")
    parser.add_argument("--head", default="mlp", choices=("linear", "mlp"))
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=14)
    parser.add_argument("--samples", type=int, default=384)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--commit", type=int, default=2)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--panel", type=int, default=200,
                        help="pixel size of the map panel; 0 draws no panel")
    parser.add_argument("--skip", type=int, default=0,
                        help="actions to run before recording starts, for "
                             "skipping an opening stretch of dialogue")
    parser.add_argument("--ms", type=int, default=140, help="per frame")
    parser.add_argument("--unknown-cost", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="media/navigation.gif")
    parser.add_argument("--device", default=default_device())
    args = parser.parse_args()

    needs_model = args.policy in ("cem", "greedy")
    if needs_model and not args.checkpoint:
        raise SystemExit(f"--policy {args.policy} needs a checkpoint")

    device = t.device(args.device)
    model = probe = lookup = None
    maps = [0, 37, 38, 40]
    if needs_model:
        checkpoint = t.load(args.checkpoint, map_location=device,
                            weights_only=False)
        model = JEPA(num_frames=checkpoint["config"]["num_frames"]).to(device)
        model.load_state_dict(checkpoint["jepa"])
        model.eval()
        fitted = coords_probe.fit(model, args.data, device, head=args.head)
        probe = fitted.to_torch(device)
        lookup = action_lookup(device)
        maps = fitted.maps
        print(f"{fitted.head} head, +/-{fitted.tile_mae:.2f} tiles held out")

    walk = build_walkmap(args.data)
    field = walk.fields(maps, TARGET, (32, 32), unknown=args.unknown_cost)
    index_of = {map_id: index for index, map_id in enumerate(maps)}
    cost_of = lambda here: field_cost(field, index_of.get(int(here[0])),
                                      here[1], here[2], args.unknown_cost)

    planned: list[int] = []
    if args.policy == "route":
        route = Route.load(args.route)
        # Same rule as collect.py: replaying a route from a different save
        # state produces a confidently wrong recording. Both routes here were
        # recorded from states/bedroom.state.
        if route.state_sha1 != state_digest(args.state):
            raise SystemExit(
                f"{args.route} was recorded against a different save state "
                f"than {args.state}. Replaying it from here would show the "
                f"buttons of one run over the screens of another. Use the "
                f"state it was recorded from."
            )
        planned = route.indices[: args.skip + args.steps]
    elif args.policy == "random":
        rng = np.random.default_rng(args.seed)
        planned = rng.integers(0, len(ACTIONS),
                               args.skip + args.steps).tolist()

    # The panel draws the raw BFS distances, not the field the planner is
    # scored on. The planner's field extrapolates a cost into unvisited ground
    # so that a half-tile read-out error does not fall off a cliff, which is
    # right for scoring and wrong for a picture: it shades the whole 32x32 grid
    # and the walkable region stops being visible at all.
    display = walk.fields(maps, TARGET, (32, 32), unknown=args.unknown_cost,
                          fill=0.0)
    crops = {map_id: crop_field(display[index], args.unknown_cost)
             for map_id, index in index_of.items()}
    trails: dict[int, list[tuple[int, int]]] = {}

    def panel_for(here):
        """The current map's field, in that map's own crop, with its trail."""
        if not args.panel or here[0] not in crops or here[1] is None:
            return None
        grid, y0, x0 = crops[here[0]]
        local = (here[0], here[1] - y0, here[2] - x0)
        target = ((TARGET[1] - y0, TARGET[2] - x0) if here[0] == TARGET[0]
                  else (-9, -9))
        trail = [(y - y0, x - x0) for (y, x) in trails.get(here[0], [])]
        return map_panel(grid, local, trail, target, args.panel, font)

    field_tensor = (t.from_numpy(field).to(device) if needs_model else None)
    env = PokeEnv(args.rom, args.state, headless=True)
    font = caption_font(args.scale)
    frames: list[Image.Image] = []
    try:
        env.reset()
        env.settle()
        here = (cur_map(env), *coords(env))
        if here[1] is not None:
            trails.setdefault(here[0], []).append((here[1], here[2]))
        if not args.skip:
            frames.append(frame_image(
                env.screen(), args.scale,
                caption(0, here, cost_of(here), None), font, panel_for(here),
            ))

        spent = 0
        predicted = None
        arrived = False
        while spent < args.skip + args.steps and not arrived:
            if planned:
                batch = [planned[spent]]
            else:
                batch, predicted = model_step(
                    model, env, probe, lookup, device, field, index_of,
                    args.policy, args.horizon, args.samples, args.iterations,
                    args.commit, args.seed, spent, field_tensor,
                )
            for action_index in batch:
                if spent >= args.skip + args.steps:
                    break
                # Captured AFTER the step, so the caption describes the action
                # that produced the frame and the tile it produced.
                env.step(action_index)
                spent += 1
                here = (cur_map(env), *coords(env))
                if here[1] is not None:
                    trails.setdefault(here[0], []).append((here[1], here[2]))
                if spent > args.skip:
                    frames.append(frame_image(
                        env.screen(), args.scale,
                        caption(spent, here, cost_of(here),
                                ACTIONS[action_index], predicted),
                        font, panel_for(here),
                    ))
                if (here[0], here[1], here[2]) == TARGET:
                    arrived = True
                    break

        print(f"{args.policy}: {spent} actions, ended at {here}, "
              f"{cost_of(here):.0f} hops from {TARGET}")
    finally:
        env.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Hold the first and last frames rather than repeating them: GIF
    # optimisation drops identical consecutive frames, so duplicating them
    # does nothing, while a per-frame duration survives.
    durations = [args.ms] * len(frames)
    durations[0] = durations[-1] = max(args.ms * 6, 800)
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True)
    size = out.stat().st_size / 1024
    print(f"wrote {out} ({len(frames)} frames, {size:.0f} KB)")


if __name__ == "__main__":
    main()
