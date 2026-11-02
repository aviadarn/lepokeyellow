"""DAgger: ask the oracle what to do from the states the clone actually reaches.

The clone fitted its 208 demonstrations to loss 0.0005 and still failed every
eval run: it walked north correctly for five decisions and then oscillated
between two tiles for four hundred actions. That is the compounding-error
failure, and it is not a bug in the fit -- one optimal trajectory contains no
frame showing the way back from a mistake, because the oracle never made one.
These are the pieces that collect the learner's own states and relabel them.
"""

import numpy as np
import pytest

from poke.games import YELLOW
from poke.rl.dagger import SnapshotProbe, Visit, spread


class StoryEnv:
    """Real enough for `story_key` and `poke.objectives.current`.

    Yellow's addresses, a position the test drives, and one event byte that
    says Oak appeared and the rival fight is over -- which is `get_parcel`,
    exactly where Part 2 starts.
    """

    game = YELLOW
    #: bits 35 (rival beaten) and 39 (oak appeared) live in event byte 4.
    STORY_BYTE = 0b1000_1000

    def __init__(self, positions, map_id=12):
        self.positions = list(positions)
        self.map_id = map_id
        self.index = 0
        self.snapshots = []
        self.restored = []

    def read_ram(self, addr):
        ram = YELLOW.ram
        y, x = self.positions[min(self.index, len(self.positions) - 1)]
        if addr == ram.cur_map:
            return self.map_id
        if addr == ram.y_coord:
            return y
        if addr == ram.x_coord:
            return x
        if addr == ram.event_flags + 4:
            return self.STORY_BYTE
        return 0

    def snapshot(self):
        self.snapshots.append(self.index)
        return ("blob", self.index)

    def restore(self, blob):
        self.restored.append(blob)
        self.index = blob[1]

    def screen(self):
        return np.zeros((144, 160, 3), dtype=np.uint8)

    def step(self, action_index):
        self.index += 1


def walker(env):
    """A policy that moves the player on every turn, and records what it saw."""
    seen = []

    def policy(guarded, objective_one_hot):
        seen.append(guarded)
        env.index += 1
        return 1

    policy.seen = seen
    return policy


def test_the_probe_snapshots_the_state_behind_each_decision():
    env = StoryEnv([(1, 1), (2, 1), (3, 1)])
    probe = SnapshotProbe(walker(env), env)
    for _ in range(3):
        probe("guarded", [0.0] * 5)
    assert [visit.blob for visit in probe.visits] == [("blob", 0), ("blob", 1), ("blob", 2)]
    assert [visit.turn for visit in probe.visits] == [0, 1, 2]


def test_a_state_the_clone_returns_to_is_snapshotted_once():
    """The measured failure is an oscillation between two tiles for four
    hundred actions. A save state is 174 KiB, so snapshotting every turn
    would hold a quarter of a gigabyte of two distinct states."""
    env = StoryEnv([(1, 1), (2, 1), (1, 1), (2, 1)])
    probe = SnapshotProbe(walker(env), env)
    for _ in range(4):
        probe("guarded", [0.0] * 5)
    assert len(probe.visits) == 2
    assert probe.turns == 4
    assert probe.revisits == 2, "the oscillation, counted"


def test_the_probe_hands_the_policy_the_guarded_env_and_nothing_else():
    """The probe is training-time machinery wrapped around a contract-bound
    policy. It holds the real env to snapshot it; the policy must never see
    that object."""
    env = StoryEnv([(1, 1), (2, 1)])
    policy = walker(env)
    probe = SnapshotProbe(policy, env)
    guarded = object()
    probe(guarded, [0.0] * 5)
    assert policy.seen == [guarded]


def test_the_probe_forwards_the_policys_action():
    env = StoryEnv([(1, 1)])
    probe = SnapshotProbe(lambda guarded, objective: 4, env)
    assert probe("guarded", [0.0] * 5) == 4


def test_every_visit_names_the_objective_it_was_taken_on():
    """The relabelling search needs to know which goal to solve for, and the
    objective is read from event flags -- the one thing the contract allows."""
    env = StoryEnv([(1, 1)])
    probe = SnapshotProbe(walker(env), env)
    probe("guarded", [0.0] * 5)
    assert probe.visits[0].objective == "get_parcel"


def test_a_visit_carries_the_frames_the_clone_saw_just_before_it():
    """A relabelled episode starts in the middle of a run. Replayed from a
    fresh GRU and a fresh frame window it is a different input than the same
    game state is at play time, where the window holds the last four frames
    and the hidden state carries four hundred oscillating steps. The visit
    keeps the learner's own preceding turns so the episode can be warmed with
    them instead."""
    env = StoryEnv([(1, 1), (2, 1), (3, 1)])
    probe = SnapshotProbe(walker(env), env, context=2)
    for _ in range(3):
        probe("guarded", [0.0] * 5)
    assert probe.visits[0].context == []
    assert len(probe.visits[2].context) == 2, "the two turns before this one"
    frames = [frame for frame, _ in probe.visits[2].context]
    assert frames[0].shape == (144, 160) and frames[0].dtype == np.uint8


def test_the_context_records_what_the_learner_pressed_not_what_it_should_have():
    env = StoryEnv([(1, 1), (2, 1)])
    probe = SnapshotProbe(lambda guarded, objective: 3, env, context=4)
    probe("guarded", [0.0] * 5)
    env.index += 1
    probe("guarded", [0.0] * 5)
    assert [action for _, action in probe.visits[1].context] == [3]


def test_each_visit_counts_the_turns_the_learner_spent_in_it():
    env = StoryEnv([(1, 1), (2, 1), (1, 1), (1, 1)])
    probe = SnapshotProbe(walker(env), env)
    for _ in range(4):
        probe("guarded", [0.0] * 5)
    spent = {visit.turn: visit.revisits for visit in probe.visits}
    assert spent == {0: 2, 1: 0}, "the learner came back to the first state twice"


def test_the_stuck_states_are_the_ones_worth_asking_the_expert_about():
    """Measured: round 2 relabelled six states and five of them were from
    turns 0-14, the prefix the clone already walks correctly. Almost every
    distinct state of a stuck rollout happens early, so spreading over them
    spends the search budget on the part that works."""
    from poke.rl.dagger import by_time_spent

    visits = [Visit(b"", (), turn, "get_parcel") for turn in (0, 4, 9, 20)]
    for visit, spent in zip(visits, (0, 0, 300, 2)):
        visit.revisits = spent
    picked = by_time_spent(visits, 2)
    assert [visit.turn for visit in picked] == [9, 20], "in turn order, stuck first"


def test_a_tie_on_time_spent_prefers_the_later_state():
    from poke.rl.dagger import by_time_spent

    visits = [Visit(b"", (), turn, "get_parcel") for turn in (0, 7)]
    assert [v.turn for v in by_time_spent(visits, 1)] == [7]


def test_asking_for_more_stuck_states_than_exist_takes_them_all():
    from poke.rl.dagger import by_time_spent

    visits = [Visit(b"", (), turn, "get_parcel") for turn in (0, 7)]
    assert [v.turn for v in by_time_spent(visits, 9)] == [0, 7]


def test_spread_takes_everything_when_fewer_states_than_asked_for():
    assert spread([1, 2, 3], 10) == [1, 2, 3]


def test_spread_includes_the_first_and_the_last():
    """An oracle search costs about five minutes, so a round relabels a
    handful of states. Both ends matter: the start proves the prefix is still
    right, the end is where the clone is stuck."""
    assert spread(list(range(9)), 3) == [0, 4, 8]


def test_spread_of_one_takes_the_first():
    assert spread([7, 8, 9], 1) == [7]


def test_spread_of_none_takes_nothing():
    assert spread([7, 8, 9], 0) == []


def test_a_visit_is_hashable_free_data():
    visit = Visit(blob=b"x", key=(1, 2), turn=3, objective="get_parcel")
    assert visit.turn == 3 and visit.objective == "get_parcel"


# ---------------------------------------------------------------- relabelling

def test_every_part_two_objective_has_an_oracle_phase_that_solves_it():
    from scripts.dagger import goal_for

    env = StoryEnv([(1, 1)])
    for key in ("get_parcel", "deliver_parcel", "enter_forest"):
        goal, allowed, buttons = goal_for(env, key)
        assert callable(goal) and allowed and buttons


def test_the_map_the_clone_is_standing_on_is_added_to_the_allowed_set():
    """`phases()` lists the maps the oracle's own route crossed. A clone that
    wandered can be standing off that list, and a search rooted on a
    disallowed map prunes every child straight back out."""
    from scripts.dagger import goal_for

    _, allowed, _ = goal_for(StoryEnv([(1, 1)], map_id=13), "get_parcel")
    assert 13 in allowed and 42 in allowed


def test_an_objective_before_part_two_has_no_phase():
    from scripts.dagger import goal_for

    with pytest.raises(ValueError) as caught:
        goal_for(StoryEnv([(1, 1)]), "get_starter")
    assert "get_starter" in str(caught.value)


def test_relabelling_restores_the_state_the_clone_reached_and_solves_from_it():
    import scripts.dagger as dagger

    env = StoryEnv([(1, 1), (2, 1)])
    visit = Visit(blob=("blob", 1), key=(), turn=7, objective="get_parcel")
    calls = {}

    class FoundResult:
        found = True
        edge_blobs = [("blob", 1)]
        edge_traces = [["up"]]
        path = ["up"]
        nodes = 11
        seconds = 1.0
        end = (12, 30, 9)

    def fake_search(env_, goal, allowed, **kwargs):
        calls["allowed"] = allowed
        return FoundResult()

    dagger.search = fake_search
    try:
        steps, result = dagger.relabel(env, visit)
    finally:
        import poke.oracle
        dagger.search = poke.oracle.search
    assert env.restored[0] == ("blob", 1), "solved from where the clone was"
    assert result.found and [step.action_name for step in steps] == ["up"]
    assert 12 in calls["allowed"]


def test_an_unsolved_relabelling_contributes_no_demonstrations():
    import scripts.dagger as dagger

    class LostResult:
        found = False
        edge_blobs: list = []
        edge_traces: list = []
        path: list = []
        nodes = 3
        seconds = 0.1
        end = None

    env = StoryEnv([(1, 1)])
    dagger.search = lambda *a, **k: LostResult()
    try:
        steps, result = dagger.relabel(env, Visit(("blob", 0), (), 0, "get_parcel"))
    finally:
        import poke.oracle
        dagger.search = poke.oracle.search
    assert steps == [] and not result.found


# --------------------------------------------------------------- the dataset

def _stand_ins(tmp_path):
    """The writer only hashes the ROM and the start state into its attrs, so
    any bytes will do -- CI has neither file, and neither may be committed."""
    rom, state = tmp_path / "rom.gbc", tmp_path / "route1.state"
    rom.write_bytes(b"rom")
    state.write_bytes(b"state")
    return str(rom), str(state)


def _write(path, episodes):
    from poke.actions import ACTIONS, to_multihot
    from poke.writer import EpisodeWriter

    rom, state = _stand_ins(path.parent)
    with EpisodeWriter(path, rom, state, 60, 0, "yellow", task="dagger_part2") as writer:
        for names, key, context in episodes:
            for name in names:
                writer.add_step(np.zeros((144, 160), dtype=np.uint8),
                                to_multihot(ACTIONS.index(name)), 0, False, None)
            writer.end_episode("dagger", key, True, -1, context)


def test_a_round_file_remembers_how_many_leading_turns_are_the_learners(tmp_path):
    """The warm-up turns carry the learner's own presses. Training must not
    score them, so the count has to survive the round trip to disk."""
    from scripts.train_clone import load_episodes

    path = tmp_path / "r1.h5"
    _write(path, [(["up", "up", "left", "right"], "get_parcel", 2)])
    (frames, actions, key, context), = load_episodes(path)
    assert key == "get_parcel" and len(actions) == 4 and context == 2


def test_an_ordinary_demonstration_file_has_no_warm_up(tmp_path):
    """scripts/collect_demos.py writes goal_step -1, which is "not recorded",
    not "two turns of context"."""
    from scripts.train_clone import load_episodes

    path = tmp_path / "demos.h5"
    _write(path, [(["up", "left"], "enter_forest", -1)])
    (_, _, _, context), = load_episodes(path)
    assert context == 0


def test_the_round_writer_lays_one_episode_out_per_relabelled_state(tmp_path):
    from scripts.dagger import ContextStep, write_round
    from scripts.collect_demos import DemoStep
    from scripts.train_clone import load_episodes

    frame = np.zeros((144, 160), dtype=np.uint8)
    steps = [ContextStep(frame, 1), DemoStep(frame=frame, action_name="up")]
    path = tmp_path / "r1.h5"
    assert write_round(path, *_stand_ins(tmp_path), [(steps, "get_parcel", 1)]) == 2
    (_, actions, key, context), = load_episodes(path)
    assert key == "get_parcel" and context == 1 and len(actions) == 2
