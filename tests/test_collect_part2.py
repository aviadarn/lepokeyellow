"""The collection plan. The recording itself needs a ROM; the plan does not."""

from scripts.collect_part2 import MILESTONE_STATES, Job, episode_plan


def test_every_milestone_state_is_covered():
    plan = episode_plan(MILESTONE_STATES, per_state=3, seed=0)
    assert {job.state for job in plan} == set(MILESTONE_STATES)


def test_each_state_gets_the_requested_number_of_episodes():
    plan = episode_plan(("a.state", "b.state"), per_state=5, seed=0)
    assert sum(job.state == "a.state" for job in plan) == 5


def test_seeds_are_unique_so_two_episodes_are_not_the_same_walk():
    plan = episode_plan(MILESTONE_STATES, per_state=4, seed=7)
    assert len({job.seed for job in plan}) == len(plan)


def test_the_plan_is_deterministic_for_a_seed():
    first = episode_plan(MILESTONE_STATES, per_state=2, seed=3)
    second = episode_plan(MILESTONE_STATES, per_state=2, seed=3)
    assert first == second


def test_both_kinds_are_collected():
    """Random alone never opens a door it has to talk through."""
    plan = episode_plan(MILESTONE_STATES, per_state=4, seed=0)
    assert {job.kind for job in plan} == {"random", "sticky"}


def test_the_forest_state_is_in_the_list():
    assert "states/oracle/forest.state" in MILESTONE_STATES
