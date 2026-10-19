"""Dataset round-trip plus goal-frame selection.

These are the seam between the emulator and the planner: the collector writes
a per-step goal flag and the planner reads goal frames back out of it. Upstream
rebuilt goal frames from `party_counts` and from episode names matching the
three Red starters, so both halves had to be replaced for Yellow.
"""

import h5py
import numpy as np
import pytest

from poke.goals import check_dataset, goal_indices, load_frame, start_index
from poke.writer import SCHEMA, EpisodeWriter


@pytest.fixture
def h5_path(tmp_path):
    rom = tmp_path / "fake.gb"
    rom.write_bytes(b"\x00" * 1024)
    state = tmp_path / "s.state"
    state.write_bytes(b"\x01" * 32)
    route = tmp_path / "r.json"
    route.write_text("{}")
    return tmp_path / "poke.h5", rom, state, route


def write_episodes(h5_path, episodes, game="yellow", task="yellow_pikachu"):
    out, rom, state, route = h5_path
    with EpisodeWriter(out, rom, state, 60, 0, game=game, task=task, route=route) as w:
        for kind, name, steps in episodes:
            goal_step = -1
            for index, goal in enumerate(steps, start=1):
                frame = np.full((144, 160), index, dtype=np.uint8)
                w.add_step(frame, np.zeros(6, np.uint8), int(goal), goal)
                if goal and goal_step < 0:
                    goal_step = index
            w.end_episode(kind, name, any(steps), -1, goal_step)
    return out


def test_attributes_record_game_task_and_route(h5_path):
    out = write_episodes(h5_path, [("demo", "demo0", [0, 0, 1])])
    with h5py.File(out, "r") as f:
        assert f.attrs["schema"] == SCHEMA
        assert f.attrs["game"] == "yellow"
        assert f.attrs["task"] == "yellow_pikachu"
        assert len(f.attrs["rom_sha1"]) == 40
        assert len(f.attrs["route_sha1"]) == 40


def test_check_dataset_rejects_a_mismatched_game_or_task(h5_path):
    out = write_episodes(h5_path, [("demo", "demo0", [0, 1])])
    check_dataset(str(out), "yellow", "yellow_pikachu")
    with pytest.raises(ValueError, match="collected on 'yellow'"):
        check_dataset(str(out), "red", "yellow_pikachu")
    with pytest.raises(ValueError, match="task 'yellow_pikachu'"):
        check_dataset(str(out), "yellow", "leave_room")


def test_goal_flags_and_party_counts_are_stored_per_step(h5_path):
    out = write_episodes(h5_path, [("demo", "demo0", [0, 0, 1, 1])])
    with h5py.File(out, "r") as f:
        assert f["goal_flags"][:].tolist() == [0, 0, 1, 1]
        assert f["episode_meta"]["goal_step"][0] == 3
        assert f["frames"].shape == (4, 144, 160, 1)


def test_start_index_is_the_first_frame_of_the_first_episode(h5_path):
    out = write_episodes(h5_path, [("demo", "d", [0, 1]), ("random", "r", [0, 0])])
    assert start_index(str(out)) == 0


def test_goal_indices_picks_the_first_goal_frame(h5_path):
    out = write_episodes(h5_path, [("demo", "d0", [0, 0, 1, 1])])
    assert goal_indices(str(out)) == [2]


def test_goal_indices_prefers_demos_over_lucky_random_episodes(h5_path):
    """A random walk that stumbles into the goal arrives in an odd screen
    state, and a goal set is only as tight as its worst member."""
    out = write_episodes(
        h5_path,
        [("random", "r0", [0, 1]), ("demo", "d0", [0, 0, 1]), ("noisy", "n0", [0, 1])],
    )
    chosen = goal_indices(str(out))
    assert 1 not in chosen  # the random episode's goal frame
    assert set(chosen) <= {4, 6}


def test_goal_indices_falls_back_to_random_when_there_is_nothing_else(h5_path):
    out = write_episodes(h5_path, [("random", "r0", [0, 1])])
    assert goal_indices(str(out)) == [1]


def test_goal_indices_honours_the_limit(h5_path):
    episodes = [("demo", f"d{i}", [0, 1]) for i in range(20)]
    out = write_episodes(h5_path, episodes)
    assert len(goal_indices(str(out), limit=4)) == 4


def test_no_successful_episode_raises_rather_than_planning_to_nowhere(h5_path):
    out = write_episodes(h5_path, [("random", "r0", [0, 0, 0])])
    with pytest.raises(RuntimeError, match="no successful episode"):
        goal_indices(str(out))


def test_load_frame_scales_to_minus_one_one(h5_path):
    out = write_episodes(h5_path, [("demo", "d", [0, 1])])
    frame = load_frame(str(out), 0)
    assert frame.shape == (1, 1, 144, 160)
    # raw value 1 -> 1/127.5 - 1
    assert abs(float(frame[0, 0, 0, 0]) - (1 / 127.5 - 1.0)) < 1e-6


def test_dataset_names_the_problem_when_no_window_fits(h5_path):
    """numpy's "need at least one array to concatenate" says nothing about
    which knob is wrong. --max-horizon is the usual cause."""
    from poke.dataset import PokeDataset

    out = write_episodes(h5_path, [("demo", "d", [0, 1, 1])])
    with pytest.raises(ValueError, match="no episode is long enough"):
        PokeDataset(out, num_frames=9)
    assert len(PokeDataset(out, num_frames=3)) == 1
