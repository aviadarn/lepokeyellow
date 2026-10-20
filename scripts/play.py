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
from poke import coords_probe
from poke import progress as progress_probe
from poke.ladder import build as build_ladder
from poke.ladder import dedupe as dedupe_ladder
from poke.ladder import load_frames as load_ladder_frames
from poke.games import resolve
from poke.goals import default_device
from poke.jepa import JEPA
from poke.maps import name as map_name
from poke.milestones import chain_from
from poke.ram import TRAINER_BATTLE, battle_kind, coords, cur_map, in_battle, party_count
from poke.waypoints import build as build_waypoints
from poke.waypoints import localise

from scripts.plan_cem import action_lookup, cem_plan

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

    def embed_frames(self, frames: np.ndarray) -> t.Tensor:
        """Encode greyscale [N, H, W] rungs into [N, D]."""
        with t.inference_mode():
            return t.cat([
                self.model.encode(
                    t.from_numpy(frame.astype(np.float32) / 127.5 - 1.0)
                    .unsqueeze(0).unsqueeze(0).to(self.device)
                )
                for frame in frames
            ])

    def run_ladder(self, rungs: t.Tensor, milestone, budget: int) -> bool:
        """Climb a ladder of subgoals towards one milestone.

        Clearing a rung is relative, not absolute. Measured on this dataset,
        latent MSE from the start frame to the first rung is 1.82 while the
        median distance between consecutive rungs is 0.0017 -- three orders of
        magnitude apart, because 20 actions of dialogue barely move the screen
        while a map transition moves it enormously. No fixed threshold can span
        that, so a rung is cleared when the distance has fallen to `approach`
        of whatever it was when that rung became current, and a per-rung action
        cap stops the climb stalling on a rung it cannot reach.

        The ladder never steps backwards: one text box resembles another, so a
        distance dip alone is not evidence of progress.
        """
        rung, spent, cleared = 0, 0, 0
        entry_distance: float | None = None
        rung_started = 0

        while spent < budget and rung < len(rungs):
            if milestone.reached(self.env, self.history):
                self.history["rungs_cleared"] = cleared
                return True
            if in_battle(self.env):
                self.step(A, "battle", milestone.key)
                spent += 1
                continue

            initial = self.observe()
            distance = float((initial - rungs[rung]).square().mean())
            if entry_distance is None:
                entry_distance = distance

            cleared_now = distance <= self.args.approach * entry_distance
            gave_up = spent - rung_started >= self.args.rung_budget
            if cleared_now or gave_up:
                cleared += int(cleared_now)
                if cleared_now or not self.args.reanchor:
                    rung += 1
                    label = "cleared" if cleared_now else "ABANDONED"
                else:
                    # Advancing by one on every abandonment marches the ladder
                    # past where the player actually is: a few in a row and
                    # every remaining rung is far away, which showed up as a
                    # flat 1.9-2.0 distance plateau for 20 rungs while the
                    # player stood in Pallet Town. Re-anchor to the rung that
                    # is genuinely closest instead.
                    #
                    # Forward-only, and never back past the furthest rung
                    # cleared: one text box resembles another, so an
                    # unconstrained nearest-rung rule would happily walk the
                    # ladder backwards on a coincidental match.
                    # Bounded, because latent distance turned out to be a poor
                    # localiser: unbounded, it anchored to rung 44 of 46 while
                    # the player was still in Pallet Town with an empty party,
                    # matching a late-route rung to within 0.3787. A signal
                    # that unreliable is not allowed to move the ladder far.
                    horizon_rungs = rungs[rung : rung + self.args.anchor_span]
                    ahead = (horizon_rungs - initial).square().mean(dim=1)
                    jump = rung + int(ahead.argmin())
                    label = (f"ABANDONED -> re-anchored to {jump + 1}"
                             if jump != rung else "ABANDONED")
                    rung = jump + 1 if jump == rung else jump
                print(f"   rung {min(rung, len(rungs))}/{len(rungs)} {label} at "
                      f"action {spent} (distance {distance:.4f} from "
                      f"{entry_distance:.4f})")
                entry_distance, rung_started = None, spent
                continue

            sequence, cost, _ = cem_plan(
                self.model, initial, rungs[rung : rung + 1],
                horizon=self.args.horizon,
                samples=self.args.samples,
                elites=self.args.elites,
                iterations=self.args.iterations,
                seed=self.args.seed + spent,
                verbose=False,
            )
            for action_index in sequence.tolist()[: self.args.commit]:
                self.step(action_index, "planner", f"{milestone.key}:rung{rung}", cost)
                spent += 1
                if milestone.reached(self.env, self.history):
                    self.history["rungs_cleared"] = cleared
                    return True

        self.history["rungs_cleared"] = cleared
        self.history["rungs_total"] = len(rungs)
        return milestone.reached(self.env, self.history)

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

    in_control = True

    def here_tile(self):
        return (cur_map(self.env), *coords(self.env))

    def has_control(self) -> bool:
        """Two buttons, one snapshot, compare screens. Side-effect free."""
        blob = self.env.snapshot()
        screens = []
        try:
            for button in ("down", "right"):
                self.env.restore(blob)
                self.env.step(ACTIONS.index(button))
                screens.append(self.env.screen()[:, :, 0].tobytes())
        finally:
            self.env.restore(blob)
        return screens[0] != screens[1]

    def run_pursuit(self, points, probe, lookup, milestone, budget: int) -> bool:
        """Project the real position onto the route, aim a little ahead, repeat.

        The three earlier controller rules all chose the next target
        independently of where the player actually was, and each failed
        differently -- stranding, failure to recover, oscillation. Here the
        target is a consequence of ground truth every step: localise on the
        route by real tile, then pursue the node `lookahead` further on.

        Nothing is "abandoned", so there is no timer to tune. Progress is the
        furthest node localised, and the run ends when that stops moving.
        """
        spent, furthest, since_furthest = 0, -1, 0
        stuck_at, stuck_for = None, 0
        # Dialogue and cutscenes ignore the d-pad. At the north edge of Pallet
        # all seven buttons give the identical result, because Oak stops the
        # player and walks them to the lab; the route's own actions there are
        # b, a, up, b, b, up. Planning a path through a scene that ignores
        # paths is hopeless, and it is the same situation as a battle, which
        # already hands off. Detect it the cheap way -- the tile has not moved
        # for a while -- and mash A, which is what advances text.
        while spent < budget:
            if milestone.reached(self.env, self.history):
                self.history["pursuit_furthest"] = furthest + 1
                return True
            if since_furthest >= self.args.stall:
                print(f"   stalled: {since_furthest} actions without passing "
                      f"node {furthest + 1}/{len(points)}")
                break
            if in_battle(self.env):
                self.step(A, "battle", milestone.key)
                spent += 1
                continue

            here = self.here_tile()
            if here == stuck_at:
                stuck_for += 1
            else:
                stuck_at, stuck_for = here, 0
            # Two ways to notice the game has taken over. The cheap one -- the
            # tile has not moved for a while -- is the default because it
            # MEASURED BETTER end to end: 22/38 nodes against 17/38 for the
            # probe below. It is the worse detector and the better policy,
            # because its incidental A-mashing advances text the planner would
            # otherwise sit through.
            if not self.args.control_probe:
                if stuck_for >= self.args.dialogue_after:
                    self.step(A, "dialogue", milestone.key)
                    spent += 1
                    since_furthest += 1
                continue_planning = stuck_for < self.args.dialogue_after
                if not continue_planning:
                    continue
            # The principled one. Two buttons from one snapshot:
            # under real control `down` and `right` give different screens,
            # while a cutscene or an open text box feeds both the same thing.
            # The "tile has not moved" heuristic this replaces missed Oak's
            # encounter entirely, because the scene WALKS the player -- the
            # tile changes the whole way through.
            elif spent % self.args.control_check == 0 or not self.in_control:
                self.in_control = self.has_control()
                if not self.in_control:
                    self.step(A, "dialogue", milestone.key)
                    spent += 1
                    since_furthest += 1
                    continue

            at = localise(points, here, floor=max(furthest, 0))
            if at is None:
                # Off the route entirely: steer at the furthest node known
                # good rather than give up, so the planner has a target.
                at = max(furthest, 0)
            # Progress means ARRIVING, not being nearest. localise returns the
            # closest node at any distance, so counting it as progress reported
            # node 22 of 38 while the player stood 8 tiles away and never
            # entered the lab at all. The projection still steers; only arrival
            # advances the score.
            arrived = points[at].reached(here, self.args.waypoint_tolerance)
            if arrived and at > furthest:
                furthest, since_furthest = at, 0
                print(f"   node {at + 1}/{len(points)} REACHED at action "
                      f"{spent}: {points[at]}")
            if at >= len(points) - 1:
                break

            target = points[min(at + self.args.lookahead, len(points) - 1)]
            # A target in another room gives the planner nothing to work with:
            # the wrong-map penalty swamps the tile distance, every plan that
            # stays in this room ties, and CEM has no gradient to follow. That
            # is exactly how pursuit stalled in the house at node 10 while the
            # next nodes were in Pallet Town. Aim at the last node of the
            # CURRENT room instead -- the door the route leaves by.
            if target.map_id != here[0]:
                exits = [
                    index for index in range(at, len(points))
                    if points[index].map_id == here[0]
                ]
                # Standing ON the exit already, aiming at it gives zero
                # distance and no gradient -- the stall at node 22, on the
                # doorstep of Oak's lab. Aim through the door instead: the
                # wrong-map penalty then rewards any plan that actually
                # crosses, which is the one action needed.
                if exits and not points[max(exits)].reached(
                    here, self.args.waypoint_tolerance
                ):
                    target = points[max(exits)]
            map_index = probe["maps"].tolist().index(target.map_id)
            initial = self.observe()
            sequence, cost, _ = cem_plan(
                self.model, initial, None,
                horizon=self.args.horizon,
                samples=self.args.samples,
                elites=self.args.elites,
                iterations=self.args.iterations,
                seed=self.args.seed + spent,
                verbose=False,
                score_fn=lambda m, i, seq: coords_probe.plan_costs(
                    m, i, seq, probe, lookup, map_index, target.y, target.x,
                    repeat=self.args.action_repeat,
                ),
            )
            committed = False
            for action_index in sequence.tolist()[: self.args.commit]:
                for _ in range(self.args.action_repeat):
                    self.step(action_index, "planner",
                              f"{milestone.key}:node{at}", cost)
                    spent += 1
                    since_furthest += 1
                    if milestone.reached(self.env, self.history):
                        self.history["pursuit_furthest"] = furthest + 1
                        return True
                    # Stop repeating the moment it stops helping: walking into
                    # a wall for the rest of the run is pure waste.
                    if self.here_tile() != here:
                        here = self.here_tile()
                    elif self.args.action_repeat > 1:
                        break
                committed = True
            if not committed:
                break

        self.history["pursuit_furthest"] = furthest + 1
        self.history["pursuit_total"] = len(points)
        return milestone.reached(self.env, self.history)

    def run_waypoints(self, points, probe, lookup, milestone, budget: int) -> bool:
        """Steer tile to tile. Arrival is read from RAM, not from the model.

        The latent ladder had to guess whether a rung was cleared and got it
        wrong in both directions. Here the two jobs are separate: whether we
        have arrived is ground truth, and only the button choice is the
        model's imagination. A failure can then only be the planner's.
        """
        index, spent, reached_count = 0, 0, 0
        since_arrival = 0
        # Ground truth makes a real stall detector possible: the furthest
        # waypoint the player has ACTUALLY stood on. Re-targeting backwards is
        # needed to recover from wandering into the wrong room, but unbounded
        # it just oscillates -- one run bounced between waypoints 14 and 15 for
        # 1,200 actions. Progress is measured on the furthest reached, and the
        # run ends when that stops moving.
        furthest, since_furthest = -1, 0
        while spent < budget and index < len(points):
            if milestone.reached(self.env, self.history):
                self.history["waypoints_reached"] = reached_count
                self.history["waypoints_furthest"] = furthest + 1
                return True
            if since_furthest >= self.args.stall:
                print(f"   stalled: {since_furthest} actions without passing "
                      f"waypoint {furthest + 1}/{len(points)}")
                break
            if in_battle(self.env):
                self.step(A, "battle", milestone.key)
                spent += 1
                continue

            here = self.here_tile()
            # Skip ahead past any later waypoint already satisfied. Safe in a
            # way the latent ladder's re-anchoring was not: this compares real
            # tiles from RAM, not a latent distance that turned out to match
            # Pallet Town to a rung inside Oak's lab.
            ahead = [
                offset
                for offset in range(min(self.args.waypoint_lookahead,
                                        len(points) - index))
                if points[index + offset].reached(here, self.args.waypoint_tolerance)
            ]
            if ahead:
                jumped = max(ahead)
                index += jumped + 1
                reached_count += 1
                since_arrival = 0
                if index - 1 > furthest:
                    furthest, since_furthest = index - 1, 0
                print(f"   waypoint {index}/{len(points)} reached at action "
                      f"{spent}: {points[index - 1]}"
                      f"{f' (skipped {jumped})' if jumped else ''}")
                continue
            target = points[index]
            if since_arrival >= self.args.waypoint_budget:
                # Do NOT just advance. Marching the target forward on every
                # abandonment is what stranded the latent ladder, and doing it
                # here was worse: by waypoint 23 the plan wanted Oak's lab
                # while the player had wandered back to the bedroom.
                #
                # Re-target to the nearest upcoming waypoint ON THE PLAYER'S
                # CURRENT MAP instead. That is sound here in a way it never was
                # in latent space, because these are real tiles read from RAM.
                # Search the WHOLE list, not just forward. Walking back into
                # the house leaves no upcoming waypoint on the current map, and
                # a forward-only search then falls through to "advance anyway"
                # -- which is the stranding bug again. Re-targeting backwards
                # is the correct recovery: it steers to a tile the route
                # actually passes through from where the player now is.
                map_id, y, x = here
                candidates = [
                    (abs(points[j].y - y) + abs(points[j].x - x), -j, j)
                    for j in range(len(points))
                    if points[j].map_id == map_id and y is not None
                ]
                nearest = min(candidates)[2] if candidates else min(index + 1,
                                                                   len(points) - 1)
                direction = ("back to" if nearest < index
                             else "on to" if nearest > index else "retry")
                print(f"   waypoint {index + 1}/{len(points)} ABANDONED at "
                      f"action {spent}: wanted {target}, at {here} "
                      f"-> {direction} {nearest + 1}")
                index = nearest
                since_arrival = 0
                continue

            map_index = probe["maps"].tolist().index(target.map_id)
            initial = self.observe()
            sequence, cost, _ = cem_plan(
                self.model, initial, None,
                horizon=self.args.horizon,
                samples=self.args.samples,
                elites=self.args.elites,
                iterations=self.args.iterations,
                seed=self.args.seed + spent,
                verbose=False,
                score_fn=lambda m, i, seq: coords_probe.plan_costs(
                    m, i, seq, probe, lookup, map_index, target.y, target.x
                ),
            )
            for action_index in sequence.tolist()[: self.args.commit]:
                self.step(action_index, "planner",
                          f"{milestone.key}:wp{index}", cost)
                spent += 1
                since_arrival += 1
                since_furthest += 1
                if milestone.reached(self.env, self.history):
                    self.history["waypoints_reached"] = reached_count
                    return True
                if target.reached(self.here_tile(), self.args.waypoint_tolerance):
                    break

        self.history["waypoints_reached"] = reached_count
        self.history["waypoints_furthest"] = furthest + 1
        self.history["waypoints_total"] = len(points)
        return milestone.reached(self.env, self.history)

    def run_progress(self, scorer, milestone, budget: int) -> bool:
        """Plan by maximising predicted route progress, no rungs involved.

        There is nothing to advance through and nothing to abandon: the
        objective is a single scalar read off each predicted state, so the loop
        is plan, commit a few actions, look at the real screen, plan again.
        """
        spent = 0
        best_progress = -float("inf")
        since_improvement = 0
        while spent < budget:
            if milestone.reached(self.env, self.history):
                return True
            if in_battle(self.env):
                self.step(A, "battle", milestone.key)
                spent += 1
                continue

            initial = self.observe()
            sequence, cost, _ = cem_plan(
                self.model, initial, None,
                horizon=self.args.horizon,
                samples=self.args.samples,
                elites=self.args.elites,
                iterations=self.args.iterations,
                seed=self.args.seed + spent,
                verbose=False,
                score_fn=scorer,
            )
            for action_index in sequence.tolist()[: self.args.commit]:
                self.step(action_index, "planner", milestone.key, cost)
                spent += 1
                if milestone.reached(self.env, self.history):
                    return True

            # observe() is [1, D], so the product is [1]; take the scalar.
            here = float(
                (self.observe().cpu().numpy() @ scorer.weights_np
                 + scorer.intercept_np).reshape(-1)[0]
            )
            if here > best_progress + self.args.progress_epsilon:
                best_progress, since_improvement = here, 0
                print(f"   progress {here:7.1f} / {scorer.length} at action {spent}")
            else:
                since_improvement += self.args.commit
                if since_improvement >= self.args.stall:
                    print(f"   stalled: {since_improvement} actions without "
                          f"progress (best {best_progress:.1f})")
                    return False
        return milestone.reached(self.env, self.history)

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
    parser.add_argument("--waypoint-budget", type=int, default=25,
                        help="actions spent reaching one waypoint before "
                             "moving on to the next")
    parser.add_argument("--controller", default="pursuit",
                        choices=("pursuit", "waypoints"),
                        help="pursuit: localise on the route by real tile and "
                             "aim ahead (no timers). waypoints: the earlier "
                             "advance/abandon rule, kept for comparison.")
    parser.add_argument("--control-probe", action="store_true",
                        help="detect cutscenes by comparing two buttons' "
                             "screens instead of by the tile standing still. "
                             "More principled, measured worse (17/38 vs 22/38) "
                             "because the cruder rule mashes A more.")
    parser.add_argument("--dialogue-after", type=int, default=10,
                        help="actions with the tile unchanged before handing "
                             "over to the scripted text masher")
    parser.add_argument("--control-check", type=int, default=3,
                        help="how often to test whether the player is actually "
                             "in control (costs two emulator actions)")
    parser.add_argument("--annotations",
                        help="route annotations from scripts/annotate_route.py; "
                             "scripted steps are excluded from waypoints")
    parser.add_argument("--action-repeat", type=int, default=1,
                        help="emulator actions per planned slot. CEM samples "
                             "slots independently, so without this a plan "
                             "draws 10 consecutive presses with probability "
                             "7^-10 and cannot propose crossing an open map.")
    parser.add_argument("--lookahead", type=int, default=2,
                        help="nodes ahead of the projection to steer at")
    parser.add_argument("--waypoint-tolerance", type=int, default=1,
                        help="tiles of slack when deciding a waypoint is "
                             "reached; the probe is only good to ~1.5 tiles "
                             "and one action can walk two")
    parser.add_argument("--waypoint-lookahead", type=int, default=4,
                        help="later waypoints checked for being already "
                             "satisfied, so overshooting does not strand the "
                             "plan")
    parser.add_argument("--waypoint-stride", type=int, default=8,
                        help="every Nth distinct tile of the demonstration "
                             "becomes a waypoint")
    parser.add_argument("--objective", default="goal",
                        choices=("goal", "progress", "coords"),
                        help="goal: minimise latent distance to a goal image "
                             "(mislocalises at range). progress: maximise "
                             "route position read off the frozen encoder.")
    parser.add_argument("--coord-fit-on", default="real",
                        choices=("real", "predicted", "both"),
                        help="what the coordinate head is fitted on. Fitting "
                             "on predicted embeddings reads them far more "
                             "accurately on average and ranks actions far "
                             "worse, which is the only thing a planner uses; "
                             "see results/nav_decisions_mlp_both.json")
    parser.add_argument("--coord-head", default="mlp", choices=("linear", "mlp"),
                        help="read-out head for the coords objective. linear "
                             "ridge was the incumbent and its 1.9-tile error "
                             "in Pallet Town is finer than nothing but coarser "
                             "than the one-tile cost field it scores against.")
    parser.add_argument("--progress-epsilon", type=float, default=1.0,
                        help="progress in actions that counts as improvement")
    parser.add_argument("--ladder", action="store_true",
                        help="climb subgoals taken from the demonstration route "
                             "instead of steering straight at the milestone")
    parser.add_argument("--data", default="episodes/yellow_pikachu.h5")
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--approach", type=float, default=0.3,
                        help="clear a rung once the distance has fallen to this "
                             "fraction of what it was on arrival (scale-free)")
    # Off by default because it measured worse, twice. Advancing by one on an
    # abandoned rung cleared 9 of 46 rungs, visited 83 tiles and reached Pallet
    # Town; re-anchoring to the nearest rung cleared 5 and then 4, and the
    # bounded version spent 855 actions oscillating between the bedroom and the
    # ground floor 21 times. Latent distance is a poor localiser, so letting it
    # choose the rung does more harm than marching blindly.
    parser.add_argument("--reanchor", action="store_true",
                        help="on an abandoned rung, jump to the nearest rung "
                             "ahead instead of the next one (measured worse)")
    parser.add_argument("--anchor-span", type=int, default=3,
                        help="rungs ahead considered when re-anchoring after an "
                             "abandoned rung")
    parser.add_argument("--rung-budget", type=int, default=30,
                        help="actions spent on one rung before abandoning it")
    parser.add_argument("--horizon", type=int, default=14)
    parser.add_argument("--commit", type=int, default=1,
                        help="actions executed per plan before re-encoding. "
                             "Measured on scripts/nav_bench.py, this is the "
                             "whole difference between crossing Pallet Town "
                             "and not: 3/3 seeds at every horizon tried with "
                             "1, and 0/3 with 2. The second action of a plan "
                             "is chosen from a predicted state, and one "
                             "emulator step can cross more than one tile.")
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

        waypoints = coord_probe = lookup = None
        if args.objective == "coords":
            fitted = coords_probe.fit(model, args.data, device,
                                      head=args.coord_head,
                                      on=args.coord_fit_on)
            coord_probe = fitted.to_torch(device)
            lookup = action_lookup(device)
            waypoints = build_waypoints(args.data, args.waypoint_stride,
                                        args.annotations)
            held_out = " held out" if fitted.head == "mlp" else " in-sample"
            print(f"coordinate probe: map id {fitted.map_accuracy:.1%} accurate "
                  f"over {len(fitted.maps)} maps, {fitted.head} position head "
                  f"on {fitted.fitted_on} embeddings "
                  f"+/-{fitted.tile_mae:.2f} tiles{held_out} "
                  f"(linear {fitted.linear_tile_mae:.2f} in-sample)")
            print(f"{len(waypoints)} waypoints at stride {args.waypoint_stride}: "
                  f"{waypoints[0]} -> {waypoints[-1]}\n")

        scorer = None
        if args.objective == "progress":
            probe = progress_probe.fit(model, args.data, device)
            lookup = action_lookup(device)
            weights, intercept = probe.torch_weights(device)

            def scorer(model_, initial_, sequences):
                return progress_probe.plan_costs(
                    model_, initial_, sequences, weights, intercept, lookup
                )

            scorer.weights_np = probe.weights
            scorer.intercept_np = probe.intercept
            scorer.length = probe.length
            print(f"progress probe: cross-validated R^2 {probe.cv_r2:.3f}, "
                  f"+/-{probe.cv_mae:.0f} actions over {probe.length}")
            print(f"  error on encoded next states   {probe.encoded_mae:6.1f} actions")
            print(f"  error on PREDICTED next states {probe.predicted_mae:6.1f} actions")
            if not probe.survives_rollout:
                raise SystemExit(
                    "the probe does not survive the predictor: its error on "
                    "predicted embeddings is no better than guessing the "
                    "middle of the route, so every plan would score "
                    "plausibly and none of it would mean anything."
                )
            print()

        rungs = None
        if args.ladder:
            ladder = dedupe_ladder(args.data, build_ladder(args.data, args.stride))
            rungs = player.embed_frames(load_ladder_frames(args.data, ladder))
            print(f"ladder: {len(ladder)} rungs from demo {ladder.episode!r} "
                  f"at stride {ladder.stride}\n")

        for milestone in chain:
            print(f"-> {milestone.key}: {milestone.description}")
            if waypoints is not None and args.controller == "pursuit":
                ok = player.run_pursuit(waypoints, coord_probe, lookup,
                                        milestone, args.budget)
            elif waypoints is not None:
                ok = player.run_waypoints(waypoints, coord_probe, lookup,
                                          milestone, args.budget)
            elif scorer is not None:
                ok = player.run_progress(scorer, milestone, args.budget)
            elif rungs is not None and milestone is chain[0]:
                ok = player.run_ladder(rungs, milestone, args.budget)
            else:
                ok = player.run_milestone(milestone, args.budget)
            spent = sum(
                1 for row in player.log
                if row["milestone"].split(":", 1)[0] == milestone.key
            )
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
                     "budget", "stall", "seed", "ladder", "stride", "approach",
                     "rung_budget", "reanchor", "anchor_span", "objective",
                     "waypoint_budget", "waypoint_stride",
                     "waypoint_tolerance", "waypoint_lookahead",
                     "controller", "lookahead", "control_check",
                     "control_probe", "dialogue_after", "action_repeat")},
        "pursuit_furthest": player.history.get("pursuit_furthest"),
        "pursuit_total": player.history.get("pursuit_total"),
        "waypoints_reached": player.history.get("waypoints_reached"),
        "waypoints_furthest": player.history.get("waypoints_furthest"),
        "waypoints_total": player.history.get("waypoints_total"),
        "rungs_cleared": player.history.get("rungs_cleared"),
        "rungs_total": player.history.get("rungs_total"),
        "milestones_reached": reached,
        "milestones_total": len(chain),
        "actions_by_controller": by_controller,
        "wall_clock_s": elapsed,
        "log": player.log,
    }, indent=2) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
