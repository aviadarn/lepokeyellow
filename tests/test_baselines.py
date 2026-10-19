import numpy as np

from poke.actions import ACTIONS
from poke.baselines import Result, constant_policy, random_policy, run_all, run_episode
from poke.games import YELLOW
from poke.tasks import get
from tests.conftest import FakeEnv


class ScriptedEnv(FakeEnv):
    """Fake emulator whose goal fires only after an exact action sequence.

    Models the thing that matters: Yellow's Pikachu needs navigation then
    dialogue, so no single repeated button reaches it.
    """

    def __init__(self, required):
        super().__init__(YELLOW, {YELLOW.ram.party_count: 0})
        self.required = [ACTIONS.index(name) for name in required]
        self.taken = []

    def reset(self):
        self.taken = []
        self.ram[YELLOW.ram.party_count] = 0
        return self.screen()

    def step(self, action_idx):
        self.taken.append(action_idx)
        if self.taken[-len(self.required):] == self.required:
            self.ram[YELLOW.ram.party_count] = 1
        return self.screen()


def test_run_episode_stops_at_the_goal():
    task = get("yellow_pikachu", "yellow")
    env = ScriptedEnv(["up", "a"])
    success, actions = run_episode(env, task, [ACTIONS.index(n) for n in ["up", "a", "b", "b"]])
    assert success and actions == 2


def test_run_episode_reports_failure_after_the_horizon():
    task = get("yellow_pikachu", "yellow")
    env = ScriptedEnv(["up", "a"])
    success, actions = run_episode(env, task, [ACTIONS.index("b")] * 5)
    assert not success and actions == 5


def test_run_episode_detects_an_already_solved_start_state():
    """A save state captured one scene too late makes every policy a winner."""
    task = get("yellow_pikachu", "yellow")
    env = ScriptedEnv([])
    env.reset()
    env.ram[YELLOW.ram.party_count] = 1
    success, actions = run_episode(env, task, [])
    # reset() clears it, so this asserts reset is what defines the start.
    assert not success and actions == 0


def test_constant_policy_cannot_solve_a_navigation_task():
    """The row the blog post leaves out of its table."""
    task = get("yellow_pikachu", "yellow")
    for name in ACTIONS:
        result = constant_policy(ScriptedEnv(["up", "a"]), task, name, horizon=40)
        assert result.successes == 0, name


def test_constant_policy_does_solve_a_mash_a_task():
    """And the reason it has to be reported: upstream's Red route is
    ["a"] * 12, so `always a` wins there."""
    task = get("red_starter", "red")
    env = ScriptedEnv(["a"] * 12)
    env.game = YELLOW  # ScriptedEnv RAM is keyed on Yellow; task is count-based
    result = constant_policy(env, task, "a", horizon=14)
    assert result.successes == 1


def test_constant_policy_reports_one_run_because_the_emulator_is_deterministic():
    task = get("yellow_pikachu", "yellow")
    result = constant_policy(ScriptedEnv(["up", "a"]), task, "a", horizon=10)
    assert result.runs == 1


def test_random_policy_is_seed_reproducible():
    task = get("yellow_pikachu", "yellow")
    kwargs = dict(horizon=12, runs=30, seed=7)
    first = random_policy(ScriptedEnv(["up", "a"]), task, **kwargs)
    second = random_policy(ScriptedEnv(["up", "a"]), task, **kwargs)
    assert (first.successes, first.mean_actions) == (second.successes, second.mean_actions)


def test_random_policy_finds_a_short_sequence_sometimes():
    task = get("yellow_pikachu", "yellow")
    result = random_policy(ScriptedEnv(["up", "a"]), task, horizon=40, runs=200, seed=0)
    assert 0 < result.successes < result.runs


def test_run_all_covers_every_action_plus_random():
    task = get("yellow_pikachu", "yellow")
    results = run_all(ScriptedEnv(["up", "a"]), task, horizon=8, runs=5, seed=0)
    assert [r.name for r in results] == [f"always {n}" for n in ACTIONS] + ["uniform random"]


def test_rate_and_row_formatting():
    result = Result("uniform random", 52, 100, 13.5)
    assert result.rate == 0.52
    assert "52/100" in result.row() and "52.0%" in result.row()


def test_rate_of_zero_runs_is_not_a_division_error():
    assert Result("none", 0, 0, 0.0).rate == 0.0
