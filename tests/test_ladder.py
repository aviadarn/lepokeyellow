import numpy as np
import pytest

from poke.ladder import build, dedupe, demo_episode, load_frames
from poke.writer import EpisodeWriter


def make(h5_path, episodes):
    out, rom, state, route = h5_path
    with EpisodeWriter(out, rom, state, 60, 0, game="yellow", task="yellow_pikachu",
                       route=route) as w:
        for episode, (kind, name, length, goal_at, pixel) in enumerate(episodes):
            for step in range(1, length + 1):
                value = pixel(step) if callable(pixel) else pixel
                w.add_step(np.full((144, 160), value, np.uint8),
                           np.zeros(6, np.uint8), 0, step >= goal_at)
            w.end_episode(kind, name, goal_at <= length, -1, goal_at)
    return out


@pytest.fixture
def h5_path(tmp_path):
    rom = tmp_path / "f.gb"; rom.write_bytes(b"\x00" * 1024)
    state = tmp_path / "s.state"; state.write_bytes(b"\x01" * 16)
    route = tmp_path / "r.json"; route.write_text("{}")
    return tmp_path / "poke.h5", rom, state, route


def test_picks_the_longest_successful_demo(h5_path):
    out = make(h5_path, [
        ("demo", "short", 10, 10, lambda s: s),
        ("demo", "long", 40, 40, lambda s: s),
        ("random", "lucky", 90, 5, lambda s: s),
    ])
    start, length, name = demo_episode(str(out))
    assert (length, name) == (40, "long")


def test_named_episode_can_be_requested(h5_path):
    out = make(h5_path, [("demo", "a", 10, 10, lambda s: s),
                         ("demo", "b", 20, 20, lambda s: s)])
    assert demo_episode(str(out), "a")[2] == "a"
    with pytest.raises(ValueError, match="no demo episode named"):
        demo_episode(str(out), "missing")


def test_no_demo_episode_is_an_error_not_an_empty_ladder(h5_path):
    out = make(h5_path, [("random", "r", 20, 99, lambda s: s)])
    with pytest.raises(RuntimeError, match="no successful demo episode"):
        build(str(out))


def test_rungs_follow_the_stride_and_always_end_on_the_goal_frame(h5_path):
    out = make(h5_path, [("demo", "d", 41, 41, lambda s: s)])
    ladder = build(str(out), stride=10)
    # Start rung dropped (already satisfied); final frame always present.
    assert ladder.indices == (10, 20, 30, 40)
    assert ladder.indices[-1] == 40


def test_final_frame_is_not_duplicated_when_the_stride_lands_on_it(h5_path):
    out = make(h5_path, [("demo", "d", 41, 41, lambda s: s)])
    ladder = build(str(out), stride=40)
    assert ladder.indices == (40,)


def test_stride_must_be_positive(h5_path):
    out = make(h5_path, [("demo", "d", 10, 10, lambda s: s)])
    with pytest.raises(ValueError, match="stride must be positive"):
        build(str(out), stride=0)


def test_dedupe_collapses_a_held_screen(h5_path):
    """Dialogue holds one screen for many actions, so a fixed stride yields runs
    of identical rungs that each have to be cleared while the planner has
    nothing to do."""
    out = make(h5_path, [("demo", "d", 61, 61, lambda s: 7 if s <= 45 else s)])
    ladder = build(str(out), stride=10)
    assert len(ladder) == 6
    assert len(dedupe(str(out), ladder)) < len(ladder)


def test_dedupe_keeps_distinct_rungs(h5_path):
    out = make(h5_path, [("demo", "d", 41, 41, lambda s: s)])
    ladder = build(str(out), stride=10)
    assert dedupe(str(out), ladder).indices == ladder.indices


def test_frames_load_in_rung_order(h5_path):
    out = make(h5_path, [("demo", "d", 41, 41, lambda s: s)])
    ladder = build(str(out), stride=10)
    frames = load_frames(str(out), ladder)
    assert frames.shape == (4, 144, 160)
    assert [int(f[0, 0]) for f in frames] == [11, 21, 31, 41]
