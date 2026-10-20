import pytest

from poke.games import RED, YELLOW
from poke.tasks import TASKS, for_game, get
from tests.conftest import FakeEnv


def test_yellow_has_no_starter_choice_task():
    """Yellow's lab holds one ball and the rival takes it, so the upstream
    three-starter task is not portable. It stays Red-only on purpose."""
    assert not get("red_starter").supports("yellow")
    assert {t.key for t in for_game("yellow")} == {"yellow_pikachu", "leave_room"}


def test_asking_for_a_task_on_the_wrong_game_raises():
    with pytest.raises(ValueError, match="is defined for"):
        get("red_starter", "yellow")
    with pytest.raises(ValueError, match="unknown task"):
        get("gym_badge")


def test_pikachu_goal_fires_on_party_count():
    task = get("yellow_pikachu", "yellow")
    env = FakeEnv(YELLOW, {YELLOW.ram.party_count: 0})
    context = task.start_context(env)
    assert not task.goal(env, context)
    env.write(YELLOW.ram.party_count, 1)
    assert task.goal(env, context)


def test_leave_room_is_relative_to_the_starting_map():
    task = get("leave_room", "yellow")
    env = FakeEnv(YELLOW, {YELLOW.ram.cur_map: 40})
    context = task.start_context(env)
    assert context["start_map"] == 40
    assert not task.goal(env, context)
    env.write(YELLOW.ram.cur_map, 0)
    assert task.goal(env, context)


def test_leave_room_needs_a_readable_map():
    """An unreadable map must not look like a transition."""
    task = get("leave_room", "yellow")
    env = FakeEnv(YELLOW)
    context = task.start_context(env)
    assert not task.goal(env, context)


def test_leave_room_works_on_both_versions():
    task = get("leave_room")
    for game in (RED, YELLOW):
        env = FakeEnv(game, {game.ram.cur_map: 12})
        context = task.start_context(env)
        env.write(game.ram.cur_map, 13)
        assert task.goal(env, context)


def test_the_pikachu_task_starts_from_a_new_game_not_the_lab():
    """A lab start state is unreachable-by-construction: Oak has not stopped
    the player north of Pallet yet, so nothing in the lab can be taken. All
    seven constant buttons and 50 random seeds scored 0 there."""
    task = get("yellow_pikachu", "yellow")
    assert task.state == "states/bedroom.state"
    # Random needs 525-2,600 actions, so a 40-step horizon would measure nothing.
    assert task.horizon >= 500


def test_every_task_declares_a_state_and_horizon():
    for task in TASKS.values():
        assert task.state.endswith(".state")
        assert task.horizon >= 1
        assert task.games
        assert task.description
