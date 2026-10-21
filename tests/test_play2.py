import random

import numpy as np

from poke.actions import ACTIONS
from poke.contract import ContractViolation
from poke.control import TEXT_BOX_INTERIOR
from poke.games import YELLOW
from scripts.play2 import play, policy_for
from tests.test_oracle import CorridorEnv


class BoxedCorridorEnv(CorridorEnv):
    """A corridor with a text box permanently drawn over it."""

    def screen(self):
        frame = super().screen()
        frame[TEXT_BOX_INTERIOR] = 255
        return frame


def test_constant_policy_presses_only_its_button():
    policy = policy_for("up", random.Random(0))
    assert all(policy(None, [0.0] * 5) == ACTIONS.index("up") for _ in range(10))


def test_random_policy_is_seeded_and_reproducible():
    first = [policy_for("random", random.Random(7))(None, [0.0] * 5) for _ in range(20)]
    second = [policy_for("random", random.Random(7))(None, [0.0] * 5) for _ in range(20)]
    assert first == second


def test_play_stops_at_the_budget():
    result = play(CorridorEnv(), policy_for("left", random.Random(0)), budget=12)
    assert result["actions"] == 12


def test_play_hands_the_policy_a_guarded_env():
    """A policy that peeks at the map must blow up, not quietly win."""
    def cheater(env, objective):
        env.read_ram(YELLOW.ram.cur_map)
        return ACTIONS.index("right")

    try:
        play(CorridorEnv(), cheater, budget=4)
    except ContractViolation as error:
        assert "position" in str(error)
    else:
        raise AssertionError("the runner handed the policy an unguarded env")


def test_handler_actions_count_against_the_budget():
    """A run that spends its whole budget in menus has navigated nowhere, and
    the number that says so is the learned share."""
    result = play(BoxedCorridorEnv(), policy_for("right", random.Random(0)),
                  budget=10)
    assert result["actions"] == 10
    assert result["by_controller"].get("policy", 0) == 0
    assert result["learned_share"] == 0.0


def test_objectives_already_true_at_the_start_are_not_credited():
    """Stage A starts on Route 1, where trigger_oak and get_starter are
    already flagged. Crediting them made all eight constant buttons look like
    they had cleared two objectives apiece."""
    env = CorridorEnv()
    # OAK_APPEARED_IN_PALLET is bit 39: byte 4, bit 7.
    env.ram[YELLOW.ram.event_flags + 39 // 8] = 1 << (39 % 8)
    result = play(env, policy_for("right", random.Random(0)), budget=4)
    assert result["already_true_at_start"] == ["trigger_oak"]
    assert result["objectives"]["trigger_oak"] is None


def test_reached_objectives_record_the_action_they_landed_on():
    """The corridor's `a` at the far end sets event byte 0, which is bit 0 --
    not one of the story bits -- so nothing should be marked reached."""
    result = play(CorridorEnv(), policy_for("right", random.Random(0)), budget=8)
    assert set(result["objectives"]) == {
        "trigger_oak", "get_starter", "get_parcel",
        "deliver_parcel", "enter_forest",
    }
    assert result["objectives"]["get_parcel"] is None
