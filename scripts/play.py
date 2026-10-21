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
from poke.walkmap import build as build_walkmap
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
from poke.waypoints import gate, localise

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
        self.blocked: set[int] = set()
        # Trigger tiles stood on, exactly. A scripted waypoint is a gate, and
        # a gate is only open once the player has been ON it.
        self.touched: set[int] = set()
        self.a_run = 0
        self.talked: dict[int, int] = {}
        self.waited: dict[int, int] = {}
        self.since_move = 0
        self.party_was = -1
        self.decline = 0
        self.faced_at = None
        # Edges the player has walked into. Four-adjacency assumes that two
        # recorded tiles side by side are connected, and inside Oak's lab a
        # counter proves otherwise -- so the assumption is held until the game
        # refutes it, and the cost field is rebuilt without the edge.
        self.walls: set[tuple[tuple, tuple]] = set()
        self.tried: set[int] = set()
        self.approached: dict[int, int] = {}
        self.visits: np.ndarray | None = None
        self.walk = None
        self.maps: list[int] = []
        self._fields: dict[tuple[int, int, int], t.Tensor] = {}

    def use_walkmap(self, walk, maps: list[int]) -> None:
        self.walk, self.maps = walk, maps
        self.visits = np.zeros((len(maps), 32, 32), dtype=np.float32)

    def note_visit(self) -> None:
        """Count the tile the player is standing on.

        An MPC planner with no memory cannot tell that it has already tried a
        direction, and two tiles that each look better from the other produce a
        clean infinite loop -- measured beside the front door, where the plan
        from (7, 1) scored 2.0 and the plan from (7, 3) scored 5.0, so the
        planner walked between them 44 times. A small cost per previous visit
        breaks the tie in favour of ground it has not already covered, and
        decays to nothing on tiles it passes through once.
        """
        if self.visits is None or not self.maps:
            return
        here = self.here_tile()
        if here[0] in self.maps and here[1] is not None:
            index = self.maps.index(here[0])
            rows, cols = self.visits.shape[1:]
            self.visits[index,
                        min(max(here[1], 0), rows - 1),
                        min(max(here[2], 0), cols - 1)] += 1.0

    def cost_field(self, target):
        """The waypoint's field plus what revisiting already costs."""
        field = self.field_for(target)
        if field is None or self.visits is None:
            return field
        penalty = t.from_numpy(
            np.minimum(self.visits, self.args.revisit_cap)
            * self.args.revisit_cost
        ).float().to(self.device)
        return field + penalty

    def field_for(self, target):
        """BFS cost to one waypoint, as [n_maps, H, W]; None without a walkmap.

        Cached per waypoint: the search is over a few hundred observed tiles
        and the route revisits the same targets, so it is built once each.
        """
        if self.walk is None:
            return None
        key = (target.map_id, target.y, target.x)
        if key not in self._fields:
            grids = self.walk.fields(self.maps, key, (32, 32),
                                     unknown=self.args.unknown_cost,
                                     fill=self.args.field_fill,
                                     blocked=frozenset(self.walls))
            self._fields[key] = t.from_numpy(grids).to(self.device)
        return self._fields[key]

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

    def allowed_actions(self) -> t.Tensor:
        """Actions not yet known to do nothing from the tile standing on."""
        return t.tensor([index not in self.blocked
                         for index in range(len(ACTIONS))])

    def step(self, action_index: int, controller: str, milestone: str, cost=None) -> None:
        before_tile = self.here_tile()
        before_screen = self.env.screen()
        self.env.step(action_index)
        # An action that moves nothing and changes nothing on screen did
        # nothing, and doing it again will also do nothing. Measured without
        # this: ten `down` presses into the wall beside the front door, then
        # fifty A presses because the dialogue rule fires on "the tile stopped
        # moving" and there was no dialogue to advance. Blocks are dropped the
        # moment the tile changes, so this is a memory of the current square
        # and not a belief about the map.
        difference = float(np.abs(
            self.env.screen().astype(np.int16) - before_screen.astype(np.int16)
        ).mean())
        if self.here_tile() != before_tile:
            self.blocked = set()
            self.tried = set()
        else:
            # A direction pressed twice from one tile without moving is a
            # wall: the first press only turns the player, the second would
            # walk if anything were open. The screen test below cannot see
            # this one -- Oak and the rival animate every frame, so the mean
            # pixel difference never falls under the threshold inside the lab,
            # and the planner pressed right into a counter 33 times without
            # ever marking it. Confirm control first, because the player also
            # stands still through a text box, and a cutscene that ignores the
            # d-pad would otherwise be recorded as walls on every side.
            # Only just after a move, because the player also stands still
            # through a text box and a cutscene ignores the d-pad, and a scene
            # would otherwise be recorded as walls on every side. Recency is
            # the test rather than `has_control`, which costs a snapshot and a
            # restore: run inside every step it made the whole run
            # irreproducible from its own action log.
            if (action_index in self.tried
                    and ACTIONS[action_index] in self.STEPS
                    and self.since_move <= self.args.wall_window):
                self.note_wall(before_tile, action_index)
            self.tried.add(action_index)
            if difference <= self.args.screen_epsilon:
                self.blocked.add(action_index)
        if self.here_tile() == before_tile:
            self.since_move += 1
            self.a_run += 1 if ACTIONS[action_index] == "a" else 0
        else:
            self.since_move = 0
            self.a_run = 0
        # A memory of what does nothing HERE goes stale when the world moves
        # on: A pressed into a closed text box is blocked, then the party gains
        # a Pikachu and the next box needs A. Measured, the planner spent 488
        # actions pressing right at one tile with A masked out of every plan.
        party = party_count(self.env)
        if party != self.party_was:
            self.party_was = party
            self.blocked = set()
            self.tried = set()
            if party:
                self.decline = self.args.decline
        self.note_visit()
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

    STEPS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

    def talk(self) -> int:
        """A to advance text, B once A has stopped getting anywhere.

        A commits and B cancels, and the difference matters exactly once in
        this story: the hand-over ends on "do you want to give a nickname?",
        where A answers YES and opens a naming screen in which A types another
        letter, B deletes one, and the way out is a button this action set does
        not have. Measured, the run sat there for its whole remaining budget.
        B alone is no answer either -- it never advances the hand-over at all,
        80 presses, no Pikachu. So: A, and a B whenever a run of A presses has
        moved nothing, because that is what a menu looks like from here and a
        conversation does not.
        """
        if self.a_run >= self.args.talk_run:
            self.a_run = 0
            return ACTIONS.index("b")
        return A

    def note_wall(self, tile, action_index: int) -> None:
        """A direction that moved nothing and drew nothing is a wall.

        Recorded as an edge of the graph rather than as a fact about the tile,
        because that is what the cost field is made of: dropping it re-runs BFS
        around the obstacle instead of through it. Dropped caches, so the next
        plan is scored on the corrected field.
        """
        step = self.STEPS.get(ACTIONS[action_index])
        if step is None or self.walk is None or tile[1] is None:
            return
        edge = (tile, (tile[0], tile[1] + step[0], tile[2] + step[1]))
        if edge not in self.walls:
            self.walls.add(edge)
            self._fields.clear()

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
        # The route is one route, not one per milestone. Starting each
        # milestone from node 0 let the projection fall back onto tiles the
        # player had already left: after the rival's fight, 207 of 336 actions
        # were spent pursuing a node four tiles BEHIND, and the lab door --
        # the only thing between the player and the next three milestones --
        # was never targeted at all.
        spent, since_furthest = 0, 0
        furthest = max(self.history.get("pursuit_furthest", 0) - 1, -1)
        aimed = max(furthest, 0)
        route_done = False
        stuck_at, stuck_for, talking_for = None, 0, 0
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
                waiting = next((index for index, point in enumerate(points)
                                if point.scripted and index not in self.touched),
                               None)
                if waiting is not None:
                    print(f"   stalled waiting for node {waiting + 1}/"
                          f"{len(points)} ({points[waiting]}); going on "
                          f"without it")
                    self.touched.add(waiting)
                    since_furthest = 0
                    continue
                print(f"   stalled: {since_furthest} actions without passing "
                      f"node {furthest + 1}/{len(points)}")
                break
            if in_battle(self.env):
                self.step(A, "battle", milestone.key)
                spent += 1
                continue

            # Getting out of the scene the party change starts. Oak carries
            # on talking, and somewhere in it comes "do you want to give a
            # nickname?", which is a yes/no box: A answers YES and opens a
            # naming screen where A types another letter, B deletes one, and
            # the way out is a button this action set does not have. Measured,
            # the run reached that screen holding a Pikachu and spent every
            # remaining action inside it -- 207 A presses at one tile, unable
            # to move.
            #
            # B is no answer either: in Gen 1 it does not advance text at all,
            # so 24 B presses left Oak mid-sentence. `down` and A in turn is
            # what works, and for one reason -- `down` does nothing to a line
            # of text and moves the cursor to NO in a box, so the A that
            # follows either turns the page or declines. Measured from the end
            # of the recorded route: the rival's fight starts 23 actions later.
            if self.decline > 0:
                self.decline -= 1
                self.step(A if self.decline % 2 else ACTIONS.index("down"),
                          "decline", milestone.key)
                spent += 1
                continue

            here = self.here_tile()
            if here == stuck_at:
                stuck_for += 1
            else:
                stuck_at, stuck_for, talking_for = here, 0, 0
            # Two ways to notice the game has taken over. The cheap one -- the
            # tile has not moved for a while -- is the default because it
            # MEASURED BETTER end to end: 22/38 nodes against 17/38 for the
            # probe below. It is the worse detector and the better policy,
            # because its incidental A-mashing advances text the planner would
            # otherwise sit through.
            # ...and it is refuted the moment A itself changes nothing on
            # screen. Without that test the branch never hands control back --
            # it only replans once the tile moves, and if A cannot move it,
            # the tile never moves. Measured: 116 of 162 actions were A
            # presses into a game that was not showing any text.
            # A cap as well, because "A changed the screen" is not evidence
            # of progress: standing on a sign, A opens the text box and A
            # closes it, so the screen changes on every press and the player
            # never moves. Measured: 116 of 162 actions were A presses at one
            # tile in Pallet Town, which is a loop the game will happily run
            # forever.
            talking = (stuck_for >= self.args.dialogue_after
                       and A not in self.blocked
                       and talking_for < self.args.dialogue_max)
            if not self.args.control_probe:
                if talking:
                    self.step(self.talk(), "dialogue", milestone.key)
                    talking_for += 1
                    spent += 1
                    since_furthest += 1
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
                    self.step(self.talk(), "dialogue", milestone.key)
                    spent += 1
                    since_furthest += 1
                    continue

            for index, point in enumerate(points):
                # Exactly, not within the tolerance the other waypoints are
                # judged by. A trigger fires on its own tile and nowhere else,
                # and (5, 7) counts as reaching (4, 7) at tolerance 1 -- which
                # is how the gate came to open on a tile the scene had never
                # fired from. A trigger that cannot be stood on is given up
                # below rather than accepted from next door.
                if (point.reached(here, 0) and index not in self.touched
                        and index <= aimed + 1 + self.args.lookahead):
                    # One node per visit, the earliest still outstanding, and
                    # only within reach of where the route actually is. Map 0
                    # (0, 10) appears twice -- Oak stops the player there on
                    # the way out of Pallet, and leaving Pallet for good is
                    # the same tile -- and his scene steps onto it, walks them
                    # south and steps onto it again. Two visits marked both
                    # nodes, so the exit counted as done before the player had
                    # a Pokemon and the run never pressed up.
                    self.touched.add(index)
                    break
            at = localise(points, here, floor=max(furthest, 0))
            if at is None:
                # Off the route entirely: steer at the furthest node known
                # good rather than give up, so the planner has a target.
                at = max(furthest, 0)
            # A trigger tile you walk onto has to be in the room you are
            # standing in, and BEHIND the route's own progress. Oak's
            # encounter leaves the player at (10, 9) in Pallet, that tile is a
            # route node, and the scene carried them there rather than letting
            # them walk -- so the exact tile is never stood on and the gate
            # held the route for 250 actions while the player was already two
            # rooms away in the lab. Releasing every gate in another room was
            # too much: standing in Pallet it released the lab's, and a
            # continuation tile beside the lab door then matched the player on
            # their way IN, which carried the route past the whole lab.
            open_gates = self.touched | {
                index for index, point in enumerate(points)
                if point.map_id != here[0] and index <= max(furthest, 0)
            }
            at = gate(points, at, open_gates)
            # And a gate is bounded in actions. One press walks two tiles, so
            # whether a given tile can be landed on at all depends on parity
            # with where the walk started: the approach above stepped back and
            # forth across (6, 5) until its budget ran out, and then the run
            # sat on the global stall for 250 more.
            held = gate(points, len(points) - 1, open_gates)
            # Counted only while the player is AT the gate, not while walking
            # to it. Measured on seed 0: the gate on Pallet's north edge was
            # held from inside Oak's lab, crossing Pallet takes about fifty
            # actions on its own, and an eighty-action budget expired with the
            # player most of the way there -- so the exit was given up and
            # nothing was left to aim at.
            near_gate = (points[held].map_id == here[0] and here[1] is not None
                         and abs(points[held].y - here[1])
                         + abs(points[held].x - here[2])
                         <= self.args.script_radius)
            if held not in open_gates and near_gate:
                self.waited[held] = self.waited.get(held, 0) + 1
                if self.waited[held] > self.args.gate_budget:
                    print(f"   node {held + 1}/{len(points)} cannot be stood "
                          f"on ({points[held]}); going on without it")
                    tile = (points[held].map_id, points[held].y, points[held].x)
                    for index in range(held + 1):
                        point = points[index]
                        if (point.map_id, point.y, point.x) == tile:
                            self.touched.add(index)
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
            # Being NEAREST to the last node is not reaching it -- the same
            # mistake as counting the nearest node as progress, in the exit
            # condition this time. Every node from 9 on is inside Oak's lab, so
            # the moment the player walked through the lab door the projection
            # snapped to the final node and the run ended on the spot, 94
            # actions in, with its score still showing node 4.
            # The last node is only done when the press it carries has
            # somewhere to happen. Leaving Pallet is `up` from (0, 10), and at
            # tolerance 1 the route reported itself finished from (1, 10), one
            # tile short: 574 fallback actions followed, with nothing left
            # aiming at the tile that is the exit.
            done_within = 0 if points[-1].facing else self.args.waypoint_tolerance
            if at >= len(points) - 1 and points[-1].reached(here, done_within):
                route_done = True
                break

            # One tile short of a trigger tile that has never been stood
            # on: step onto it by the recorded coordinate instead of by the
            # plan. A trigger needs the EXACT tile -- (4, 7) fires the
            # hand-over and (5, 7) one south of it does nothing -- and a
            # read-out accurate to a quarter of a tile cannot be asked for
            # exactness. Measured without this: the planner brought the player
            # to (5, 7), the waypoint counted as reached at tolerance 1, and
            # then 125 planner actions went left and right along row 5 and
            # never once pressed up. Logged as its own controller, because
            # these actions are not the planner's.
            # Only the gate node, never any nearby scripted one. The tiles a
            # scene CARRIES the player to are scripted waypoints too, and
            # aiming at the closest of those walked the player south out of
            # (3, 5) into a wall eleven times: (4, 5) is one tile away and is
            # only in the route because the hand-over put the player there.
            # The gate node and no other. The tiles a scene CARRIES the
            # player to are scripted waypoints as well, and aiming at whichever
            # of them is nearest walked the player south out of (3, 5) into a
            # counter eleven times, and once spent the continuation's own last
            # tile before the route had even started.
            gated = (at if points[at].scripted and at not in self.touched
                     and points[at].map_id == here[0] and here[1] is not None
                     else None)
            short = (gated if gated is not None
                     and 0 < abs(points[gated].y - here[1])
                     + abs(points[gated].x - here[2])
                     <= self.args.script_approach else None)
            if (short is not None
                    and self.approached.get(short, 0) < self.args.script_steps):
                tries = self.approached.get(short, 0)
                self.approached[short] = tries + 1
                dy = points[short].y - here[1]
                dx = points[short].x - here[2]
                straight = (("up" if dy < 0 else "down") if dy
                            else ("left" if dx < 0 else "right"))
                # One press walks one tile or two, depending on what is beyond,
                # so whether a tile can be landed on at all depends on the side
                # it is approached from. (4, 7) can only be stepped onto from
                # (5, 7): the table to its north stops that walk after one
                # tile, while along row 4 a press goes straight past it to
                # (4, 8). Measured on seed 2, the approach pressed right and
                # left across it until its budget ran out, the hand-over never
                # fired, and the run finished 0/4. So after half the budget,
                # step off the line and come at it from the other axis.
                across = ((("left" if dx < 0 else "right") if dx else "down")
                          if dy else "down")
                name = across if tries >= self.args.script_steps // 2 else straight
                self.step(ACTIONS.index(name), "approach",
                          f"{milestone.key}:node{short}")
                spent += 1
                since_furthest += 1
                continue
            # A gate the player cannot get onto is not a gate. The tiles a
            # scene carries the player to are marked scripted as well, and one
            # of them -- (4, 5), where the hand-over leaves you -- has a
            # counter between it and anywhere you can walk from. Waiting for
            # it held the whole route behind a tile that cannot be stood on.
            if (short is not None
                    and self.approached.get(short, 0) >= self.args.script_steps):
                print(f"   node {short + 1}/{len(points)} is not reachable on "
                      f"foot ({points[short]}); going on without it")
                # That node and any earlier one on the same tile, never a
                # later one: giving up on a tile says nothing about a visit
                # the route has not reached yet.
                tile = (points[short].map_id, points[short].y, points[short].x)
                for index in range(short + 1):
                    point = points[index]
                    if (point.map_id, point.y, point.x) == tile:
                        self.touched.add(index)

            # Standing ON a trigger tile: turn the way the demonstration
            # turned, then talk to whatever is in front of it. A is not an
            # interaction on its own -- Gen 1 talks to the tile the player
            # FACES, and nothing in a stream of A presses ever changes that.
            # Measured on Pikachu's table: reached facing east, 60 A presses
            # do nothing; one `up` first and the party holds a Pikachu 34
            # presses later. Keyed off the tile underfoot rather than the node
            # being pursued, because standing on a node makes it "reached" and
            # the lookahead then aims past the very tile that matters.
            on = next((index for index, point in enumerate(points)
                       if point.scripted and point.facing is not None
                       and index >= max(furthest, 0)
                       and point.reached(here, 0)), None)
            if on is not None and self.talked.get(on, 0) < self.args.script_talk:
                self.talked[on] = self.talked.get(on, 0) + 1
                if self.faced_at != here:
                    self.faced_at = here
                    self.step(ACTIONS.index(points[on].facing), "script",
                              f"{milestone.key}:face{on}")
                else:
                    self.step(self.talk(), "script", f"{milestone.key}:talk{on}")
                spent += 1
                since_furthest += 1
                continue

            aimed = max(aimed, at)
            ahead = min(at + self.args.lookahead, len(points) - 1)
            # A scripted node is a gate, not a suggestion. Looking two nodes
            # ahead from (4, 10) aimed at the lab door and skipped the tile
            # that starts Oak's encounter entirely -- and walking into the lab
            # without that scene means there is no Oak inside to talk to, so
            # the run ends standing in an empty room pressing A. Never aim past
            # a scripted node that has not been stood on yet.
            for index in range(at, ahead + 1):
                if points[index].scripted and not points[index].reached(
                    here, self.args.waypoint_tolerance
                ):
                    ahead = index
                    break
            # And a gate in THIS room is the only tile in it that matters.
            # Between the lab door and Pikachu's table the demonstration spent
            # 160 actions wandering the west wall, and those tiles are route
            # nodes like any other: pursued two at a time, the planner has to
            # repeat a random walk before it is allowed to approach the
            # trigger. The cost field knows the way to any observed tile, so
            # aim at the gate and let it route.
            if self.args.aim_gate:
                gate_here = next(
                    (index for index in range(at, len(points))
                     if points[index].scripted and index not in open_gates
                     and points[index].map_id == here[0]), None)
                if gate_here is not None:
                    ahead = gate_here
            target = points[ahead]
            # A target in another room used to give the planner nothing to work
            # with: the wrong-map penalty swamped the tile distance, every plan
            # that stayed in this room tied, and CEM had no gradient. The
            # workaround was to aim at the last node of the CURRENT room
            # instead, which fails whenever the route has no further node here
            # -- measured, the bedroom holds exactly one waypoint at stride 8,
            # so after reaching it the planner aimed into another map and
            # oscillated between two tiles for 60 actions.
            #
            # The cost field fixes it properly. It is BFS over observed
            # transitions ACROSS doorways, so a tile in this room has a real
            # distance to a target in another one and no penalty is needed.
            field = self.cost_field(target)
            if field is None and target.map_id != here[0]:
                exits = [
                    index for index in range(at, len(points))
                    if points[index].map_id == here[0]
                ]
                if exits and not points[max(exits)].reached(
                    here, self.args.waypoint_tolerance
                ):
                    target = points[max(exits)]
            # Close to a tile the demonstration lost control on, alternate
            # planning with A. Oak's encounter fires at the north edge, walks
            # the player back south and fires again, so the tile never stops
            # changing and the stuck-based dialogue rule never notices -- the
            # planner reads a cutscene as navigation and tries to walk through
            # it. Alternating keeps the walking that gets there and adds the
            # text-advancing that gets through it.
            if (target.scripted
                    and target.reached(here, self.args.script_radius)
                    and spent % 2 == 0):
                self.step(self.talk(), "script", f"{milestone.key}:node{at}")
                spent += 1
                since_furthest += 1
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
                    m, i, seq, probe, lookup, map_index, target.y, target.x,
                    repeat=self.args.action_repeat, field=field,
                ),
                allowed=self.allowed_actions(),
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

        # The route ends where the demonstration ended: standing in front of
        # Oak, not holding a Pikachu. Receiving it is dialogue, and the walking
        # is finished, so whatever budget is left goes on advancing text.
        # Without this the run reported "failed" having completed every one of
        # its waypoints, which is the difference between a navigation result
        # and a milestone.
        while route_done and spent < budget:
            if milestone.reached(self.env, self.history):
                break
            if in_battle(self.env):
                self.step(A, "battle", milestone.key)
            elif self.args.fallback == "a":
                # A alone is not an interaction: Gen 1 only talks to what the
                # player is FACING, and mashing A never changes which way that
                # is. Measured, 489 A presses standing on the route's last tile
                # in Oak's lab with nothing happening. Turn, then talk.
                # Turn, talk, then refuse. B matters as much as A here: the
                # hand-over ends on "do you want to give a nickname?", which
                # is a yes/no box, and A answers YES. That opens the naming
                # screen, where A types another letter and the run never gets
                # out -- measured, 200 A presses on a full name box with the
                # player unable to move. B answers no and backs out of a menu.
                turn = (spent // 3) % 4
                self.step((ACTIONS.index(("up", "right", "down", "left")[turn]),
                           A, ACTIONS.index("b"))[spent % 3],
                          "fallback", f"{milestone.key}:route-end")
            else:
                self.step(int(t.randint(0, len(ACTIONS), (1,)).item()),
                          "fallback", f"{milestone.key}:route-end")
            spent += 1

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
                    m, i, seq, probe, lookup, map_index, target.y, target.x,
                    field=self.cost_field(target),
                ),
                allowed=self.allowed_actions(),
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


def continuation(spec: str, offset: int) -> list:
    """Extra tiles to walk after the demonstration stops.

    The recorded route ends on the frame Pikachu is handed over, so every
    milestone after the first is past the end of the data. These are tiles the
    walkmap already holds -- the lab's door and the top of Pallet -- and the
    press each one needs to go through: a door is a step, not a conversation,
    and the top edge of a map is the same.
    """
    from poke.waypoints import Waypoint

    points = []
    for index, chunk in enumerate(part for part in spec.split(";") if part.strip()):
        fields = chunk.split(",")
        if len(fields) not in (3, 4):
            raise SystemExit(f"--continue-to: cannot read {chunk!r}; "
                             "expected map,y,x[,press]")
        map_id, y, x = (int(value) for value in fields[:3])
        press = fields[3].strip() if len(fields) == 4 else None
        if press is not None and press not in ("up", "down", "left", "right"):
            raise SystemExit(f"--continue-to: {press!r} is not a direction")
        points.append(Waypoint(map_id=map_id, y=y, x=x,
                               route_step=offset + index,
                               scripted=press is not None, facing=press))
    return points


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
    parser.add_argument("--continue-to",
                        default="40,11,5,down;0,12,11;0,0,10,up",
                        help="tiles to walk after the demonstration's route "
                             "runs out, as map,y,x[,press] separated by ';'. "
                             "The recorded route ends the moment Pikachu is "
                             "handed over, and three of the four milestones "
                             "are on the other side of that: the rival fight "
                             "fires on the way to the lab door, and leaving "
                             "Pallet is a tile at the top of the map. The "
                             "middle one is the first tile outside the lab "
                             "door that the dataset actually recorded: the "
                             "door drops the player on a tile it never saw, "
                             "and steering straight for the top of the map "
                             "from there followed extrapolated cost into "
                             "Pallet's east fence for 228 actions.")
    parser.add_argument("--aim-gate", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="pursue the next trigger tile in this room "
                             "directly, instead of the demonstration's own "
                             "wandering between here and it")
    parser.add_argument("--decline", type=int, default=24,
                        help="actions spent turning the page after the party "
                             "changes, alternating down and A, so the nickname "
                             "prompt is answered no rather than yes")
    parser.add_argument("--wall-window", type=int, default=8,
                        help="a repeated direction counts as a wall only this "
                             "many actions after the player last moved; a "
                             "cutscene ignores the d-pad for far longer")
    parser.add_argument("--gate-budget", type=int, default=80,
                        help="actions the route may wait for one trigger tile "
                             "before going on without it")
    parser.add_argument("--talk-run", type=int, default=8,
                        help="A presses that move nothing before one is a B. "
                             "A menu eats A presses forever; a conversation "
                             "does not.")
    parser.add_argument("--script-approach", type=int, default=2,
                        help="step onto a trigger tile by its recorded "
                             "coordinate from this many tiles away. A trigger "
                             "needs the exact tile; the planner's read-out is "
                             "accurate to a quarter of one.")
    parser.add_argument("--script-steps", type=int, default=20,
                        help="actions the coordinate approach may spend per "
                             "trigger tile before handing back to the planner")
    parser.add_argument("--script-talk", type=int, default=48,
                        help="actions spent turning and talking while standing "
                             "on a trigger tile. Pikachu's table needs 34 A "
                             "presses after the turn.")
    parser.add_argument("--script-radius", type=int, default=5,
                        help="tiles from a scripted waypoint within which the "
                             "planner alternates with A to advance a scene")
    parser.add_argument("--revisit-cost", type=float, default=0.75,
                        help="hops of cost added per previous visit to a tile")
    parser.add_argument("--revisit-cap", type=float, default=8.0,
                        help="most visits a tile can be charged for")
    parser.add_argument("--screen-epsilon", type=float, default=0.1,
                        help="mean pixel change below which an action is "
                             "treated as having done nothing at all")
    parser.add_argument("--dialogue-max", type=int, default=20,
                        help="A presses at one tile before the dialogue "
                             "hypothesis is abandoned and planning resumes")
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
    parser.add_argument("--walkmap", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="score plans on BFS hops over observed "
                             "transitions, across doorways, instead of "
                             "Manhattan distance plus a wrong-map penalty")
    parser.add_argument("--unknown-cost", type=float, default=120.0)
    parser.add_argument("--field-fill", type=float, default=6.0,
                        help="cost per tile of extrapolation beyond the "
                             "recorded region. It has to outrun the revisit "
                             "penalty: at 2.0 a tile costs less than three "
                             "visits do, so the tabu that breaks loops inside "
                             "the rooms also drives the player out of the data "
                             "-- measured, 226 actions against Pallet's east "
                             "fence on ground no episode ever walked. Turning "
                             "the tabu down instead cost the lab: the loop by "
                             "the door went unbroken and the run never left.")
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
            waypoints = waypoints + continuation(args.continue_to,
                                                 len(waypoints))
            held_out = " held out" if fitted.head == "mlp" else " in-sample"
            print(f"coordinate probe: map id {fitted.map_accuracy:.1%} accurate "
                  f"over {len(fitted.maps)} maps, {fitted.head} position head "
                  f"on {fitted.fitted_on} embeddings "
                  f"+/-{fitted.tile_mae:.2f} tiles{held_out} "
                  f"(linear {fitted.linear_tile_mae:.2f} in-sample)")
            print(f"{len(waypoints)} waypoints at stride {args.waypoint_stride}: "
                  f"{waypoints[0]} -> {waypoints[-1]}")
            if args.walkmap:
                walk = build_walkmap(args.data)
                player.use_walkmap(walk, fitted.maps)
                print(f"walkmap: {len(walk.nodes())} tiles across "
                      f"{len(fitted.maps)} maps, {len(walk.doors)} doorways")
            print()

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
        "walls": sorted(f"{a}->{b}" for a, b in player.walls),
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
