import numpy as np

from poke.actions import ACTIONS
from poke.control import TEXT_BOX_INTERIOR
from poke.games import YELLOW
from poke.handlers import HandlerState, choose
from poke.ram import NO_BATTLE, TRAINER_BATTLE, WILD_BATTLE
from tests.conftest import FakeEnv

A = ACTIONS.index("a")
DOWN = ACTIONS.index("down")
NOOP = ACTIONS.index("noop")


class ScreenEnv(FakeEnv):
    def __init__(self, box=False, battle=NO_BATTLE, party=1):
        super().__init__(YELLOW, {
            YELLOW.ram.in_battle: battle,
            YELLOW.ram.party_count: party,
        })
        self.frame = np.zeros((144, 160, 3), dtype=np.uint8)
        if box:
            self.frame[TEXT_BOX_INTERIOR] = 255

    def screen(self):
        return self.frame


def test_free_overworld_turn_belongs_to_the_policy():
    assert choose(ScreenEnv(), HandlerState()) is None


def test_an_open_box_is_advanced_with_a():
    action, who = choose(ScreenEnv(box=True), HandlerState())
    assert (action, who) == (A, "text")


def test_a_battle_is_mashed_with_a():
    for battle in (WILD_BATTLE, TRAINER_BATTLE):
        action, who = choose(ScreenEnv(battle=battle), HandlerState())
        assert (action, who) == (A, "battle")


def test_battle_outranks_text_so_a_fight_is_never_mislabelled():
    _, who = choose(ScreenEnv(box=True, battle=WILD_BATTLE), HandlerState())
    assert who == "battle"


def test_warp_emits_one_noop_before_anything_else():
    """The first input after a warp is swallowed by the fade-in: the state does
    not change, and every 'no effect means wall' rule then reads the door it
    just walked through as a dead end. One no-op gives control back."""
    state = HandlerState()
    state.warped = True
    action, who = choose(ScreenEnv(), state)
    assert (action, who) == (NOOP, "no_control")
    assert not state.warped
    assert choose(ScreenEnv(), state) is None


def test_a_long_run_of_a_switches_to_down_a_to_decline():
    """A answers YES to 'give a nickname?', and inside the naming screen A types
    a letter and B deletes one. The way past is down then A, alternating."""
    state = HandlerState(a_run=8, decline_left=4)
    first, who = choose(ScreenEnv(box=True), state)
    assert (first, who) == (DOWN, "decline")
    second, _ = choose(ScreenEnv(box=True), state)
    assert second == A


def test_decline_runs_out_and_hands_back_to_plain_text():
    """Driven the way the runner drives it: every chosen action is fed back
    through `note`. That matters here -- the decline pattern presses `down`
    every other turn, which is what lets the A-run decay and stops the budget
    from re-arming itself forever the moment it empties."""
    env, state = ScreenEnv(box=True), HandlerState(a_run=8)
    seen = []
    for _ in range(24):
        action, who = choose(env, state)
        seen.append(who)
        state.note(action, moved_map=False)
    assert set(seen) == {"decline"}
    assert state.decline_left == 0
    action, who = choose(env, state)
    assert (action, who) == (A, "text")


def test_handlers_never_choose_a_direction_in_a_battle():
    state = HandlerState(a_run=99, decline_left=9)
    action, who = choose(ScreenEnv(battle=WILD_BATTLE), state)
    assert (action, who) == (A, "battle")


def test_note_arms_the_warp_no_op_and_tracks_the_a_run():
    state = HandlerState()
    state.note(A, moved_map=False)
    state.note(A, moved_map=False)
    assert state.a_run == 2
    assert not state.warped
    state.note(DOWN, moved_map=True)
    assert state.a_run == 0
    assert state.warped
