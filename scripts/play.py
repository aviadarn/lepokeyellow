"""Play the game: closed-loop planning along a chain of subgoals.

Everything else in this repo plans once, open-loop, to one goal -- which is
what the blog post does, and it is why a 14-action plan is the ceiling. Latent
rollouts compound error, so a 40-action blind plan is hopeless no matter how
good the model is.

This is the loop that makes a world model usable instead:

    plan H actions, execute the first few, RE-ENCODE THE REAL SCREEN, plan again

Three controllers share the pad, and every action records which one chose it:

  planner  CEM in latent space, steering towards the current milestone's goal
           image (the screen of the bootstrap state for that point in the story)
  battle   a scripted policy while wIsInBattle is set. The planner has no
           business in a battle menu and the fight is unavoidable -- the rival
           challenges the player the moment Oak hands over Pikachu. Mashing A
           cleared it on the cartridge in 43 presses with Pikachu intact.
  fallback for milestones with no goal image yet, so the run reports "no goal
           image" rather than quietly steering at nothing

  uv run python -m scripts.play checkpoints/yellow_pikachu/best.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch as t

from poke.actions import ACTIONS
from poke.env import PokeEnv
from poke.games import resolve
from poke.goals import default_device
from poke.jepa import JEPA
from poke.maps import name as map_name
from poke.milestones import chain_from
from poke.ram import TRAINER_BATTLE, battle_kind, coords, cur_map, in_battle, party_count

from scripts.plan_cem import cem_plan

A = ACTIONS.index("a")


def to_tensor(frame: np.ndarray, device) -> t.Tensor:
    """Frame -> model input. Channel 0 is luminance because the env forces DMG."""
    pixels = frame[:, :, 0].astype(np.float32) / 127.5 - 1.0
    return t.from_numpy(pixels).unsqueeze(0).unsqueeze(0).to(device)


class Player:
    def __init__(self, model, env, device, args):
        self.model, self.env, self.device, self.args = model, env, device, args
        self.log: list[dict] = []
        self.history: dict = {}

    def observe(self) -> t.Tensor:
        with t.inference_mode():
            return self.model.encode(to_tensor(self.env.screen(), self.device))

    def goal_embedding(self, state_path: str) -> t.Tensor:
        frame = self.env.screen_of_state(state_path)
        with t.inference_mode():
            return self.model.encode(to_tensor(frame, self.device))

    def step(self, action_index: int, controller: str, milestone: str, cost=None) -> None:
        self.env.step(action_index)
        if battle_kind(self.env) == TRAINER_BATTLE:
            self.history["saw_trainer_battle"] = True
        y, x = coords(self.env)
        self.log.append({
            "n": len(self.log) + 1,
            "milestone": milestone,
            "controller": controller,
            "action": ACTIONS[action_index],
            "map": cur_map(self.env),
            "y": y, "x": x,
            "party": party_count(self.env),
            "battle": battle_kind(self.env),
            "predicted_cost": cost,
        })

    def run_milestone(self, milestone, budget: int) -> bool:
        goal = (
            self.goal_embedding(milestone.goal_state)
            if milestone.is_plannable() else None
        )
        if goal is None:
            print(f"  no goal image; falling back to {self.args.fallback}")

        spent = 0
        last_progress, best_signature = 0, None
        while spent < budget:
            if milestone.reached(self.env, self.history):
                return True

            # A battle is not a navigation problem. Hand over the pad.
            if in_battle(self.env):
                self.step(A, "battle", milestone.key)
                spent += 1
                last_progress = spent
                continue

            if goal is not None:
                initial = self.observe()
                sequence, cost, _ = cem_plan(
                    self.model, initial, goal,
                    horizon=self.args.horizon,
                    samples=self.args.samples,
                    elites=self.args.elites,
                    iterations=self.args.iterations,
                    seed=self.args.seed + spent,
                    verbose=False,
                )
                # Execute only the first few, then look at the world again.
                # This is the whole point: the model never has to predict more
                # than `commit` steps ahead before being corrected.
                for action_index in sequence.tolist()[: self.args.commit]:
                    self.step(action_index, "planner", milestone.key, cost)
                    spent += 1
                    if milestone.reached(self.env, self.history):
                        return True
            else:
                index = A if self.args.fallback == "a" else int(
                    np.random.default_rng(self.args.seed + spent).integers(0, len(ACTIONS))
                )
                self.step(index, "fallback", milestone.key)
                spent += 1

            signature = (cur_map(self.env), *coords(self.env), party_count(self.env))
            if signature != best_signature:
                best_signature, last_progress = signature, spent
            elif spent - last_progress >= self.args.stall:
                print(f"  stalled: {self.args.stall} actions with no change at "
                      f"{signature}")
                return False
        return milestone.reached(self.env, self.history)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("--rom", default="pokeyellow.gbc")
    parser.add_argument("--state", default="states/bedroom.state")
    parser.add_argument("--from-milestone")
    parser.add_argument("--horizon", type=int, default=14)
    parser.add_argument("--commit", type=int, default=3,
                        help="actions executed per plan before re-encoding")
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--elites", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--budget", type=int, default=400, help="actions per milestone")
    parser.add_argument("--stall", type=int, default=60)
    parser.add_argument("--fallback", default="a", choices=("a", "random"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--act-frames", type=int, default=60)
    parser.add_argument("--out", default="results/play.json")
    parser.add_argument("--game")
    parser.add_argument("--device", default=default_device())
    args = parser.parse_args()

    game = resolve(args.rom, args.game)
    device = t.device(args.device)
    checkpoint = t.load(args.checkpoint, map_location=device, weights_only=False)
    model = JEPA(num_frames=checkpoint["config"]["num_frames"]).to(device)
    model.load_state_dict(checkpoint["jepa"])
    model.eval()

    # Every goal image this run needs, checked before the emulator starts. The
    # files are ROM-derived so tests cannot assert they exist; this is where a
    # missing one has to be caught, and it has to be here rather than at the
    # milestone that needs it, 300 actions in.
    chain = chain_from(args.from_milestone)
    missing = [
        m.goal_state for m in chain
        if m.is_plannable() and not Path(m.goal_state).exists()
    ]
    if missing:
        raise SystemExit(
            "missing goal state(s): " + ", ".join(missing) + "\nProduce them with "
            + " ".join(f"\n  uv run python -m scripts.bootstrap --to {Path(p).stem}"
                       for p in missing)
        )

    env = PokeEnv(args.rom, args.state, headless=True,
                  act_frames=args.act_frames, game=args.game)
    player = Player(model, env, device, args)
    started = time.perf_counter()
    reached: list[str] = []
    try:
        env.reset()
        print(f"game={game.key} start={args.state} at {cur_map(env)} "
              f"({map_name(cur_map(env))}), party={party_count(env)}")
        print(f"replan every {args.commit} of {args.horizon} planned actions\n")

        for milestone in chain:
            print(f"-> {milestone.key}: {milestone.description}")
            ok = player.run_milestone(milestone, args.budget)
            spent = sum(1 for row in player.log if row["milestone"] == milestone.key)
            print(f"   {'REACHED' if ok else 'failed'} after {spent} actions "
                  f"(map {cur_map(env)}, party {party_count(env)})\n")
            if not ok:
                break
            reached.append(milestone.key)
    finally:
        env.close()

    elapsed = time.perf_counter() - started
    by_controller = {}
    for row in player.log:
        by_controller[row["controller"]] = by_controller.get(row["controller"], 0) + 1

    print(f"milestones reached: {len(reached)}/{len(chain)}"
          f"  {reached}")
    print(f"actions by controller: {by_controller}")
    print(f"{elapsed:.1f}s for {len(player.log)} actions")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "game": game.key,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "start_state": args.state,
        "planner": {k: getattr(args, k) for k in
                    ("horizon", "commit", "samples", "elites", "iterations",
                     "budget", "stall", "seed")},
        "milestones_reached": reached,
        "milestones_total": len(chain),
        "actions_by_controller": by_controller,
        "wall_clock_s": elapsed,
        "log": player.log,
    }, indent=2) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
