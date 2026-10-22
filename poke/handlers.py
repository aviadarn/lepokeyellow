"""The declared scripted harness.

The play-time contract says navigation and the decision to talk are learned,
and that four things are not: advancing a box that is already open, declining
(no nickname, no purchases), fighting, and the swallowed input after a warp.
Those live here, in one file, so that "what is scripted" is a thing you can
read rather than a thing you have to trace.

Each handler returns an action index or None. `choose` runs them in priority
order and returns None when the turn belongs to the policy. Every action it
returns carries the handler's name, which is what lands in the log and what
the learned share is computed against.
"""

from __future__ import annotations

from dataclasses import dataclass

from poke.actions import ACTIONS
from poke.control import text_box_open
from poke.ram import in_battle

A = ACTIONS.index("a")
DOWN = ACTIONS.index("down")
NOOP = ACTIONS.index("noop")

# How many A presses in a row mean A is not getting anywhere. Measured in
# Part 1: the nickname screen swallowed 207 of them.
TALK_RUN = 8
# How many down/A pairs to spend getting out of a menu once that happens.
DECLINE_BUDGET = 24
# How many consecutive box turns the handler may own before it yields one to
# the policy. The text detector is a measured approximation, and a false
# positive on a walkable tile is absorbing if the handler never lets go: only
# the policy can walk off the tile, and it never gets asked. Measured before
# this existed -- the detector read a box on Pallet Town's sand at (0, 4, 3)
# and one run spent 2714 of its 4000 actions on that square.
BOX_PATIENCE = 32


@dataclass
class HandlerState:
    """What the handlers remember between turns. Reset per episode."""

    warped: bool = False
    a_run: int = 0
    decline_left: int = 0
    box_turns: int = 0

    def note(self, action_index: int, moved_map: bool) -> None:
        """Called after each action with whether the map id changed."""
        self.a_run = self.a_run + 1 if action_index == A else 0
        if moved_map:
            self.warped = True


def choose(env, state: HandlerState) -> tuple[int, str] | None:
    """The action a handler claims this turn, or None to let the policy act."""
    if in_battle(env):
        # Mashing A selects the first move, which is Thundershock. A blackout
        # is not special-cased: the run continues from wherever the game puts
        # the player and pays for it in the action count.
        return A, "battle"
    if state.warped:
        # The fade-in eats exactly one input and leaves the state unchanged.
        state.warped = False
        return NOOP, "no_control"
    if not text_box_open(env):
        state.box_turns = 0
        return None
    state.box_turns += 1
    if state.box_turns > BOX_PATIENCE:
        # Hand one turn back. A real dialogue loses a single A press and the
        # next turn is the handler's again; a tile the detector is wrong about
        # becomes walkable instead of a trap.
        state.box_turns = 0
        return None
    if state.decline_left <= 0 and state.a_run >= TALK_RUN:
        state.decline_left = DECLINE_BUDGET
    if state.decline_left > 0:
        state.decline_left -= 1
        # down then A answers NO to a yes/no box; inside a naming screen the
        # same pair walks the cursor off the letter grid to END.
        return (DOWN if state.decline_left % 2 else A), "decline"
    return A, "text"
