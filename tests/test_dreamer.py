import numpy as np
import pytest

from poke.dreamer import (ACTIONS, HOLD_TICKS, PASS, POKEDREAMER_KEYS, SCHEMA,
                          TICKS_PER_ACTION, EpisodeBuffer, RewardTracker, act,
                          check, pack, total_reward)


def test_action_indices_are_pokedreamers():
    # 0=down, 1=left, 2=right, 3=up, 4=a, 5=b, 6=start, 7=pass (their README).
    assert ACTIONS == ("down", "left", "right", "up", "a", "b", "start", None)
    assert PASS == 7


class FakePyBoy:
    def __init__(self):
        self.calls = []

    def button_press(self, button):
        self.calls.append(("press", button))

    def button_release(self, button):
        self.calls.append(("release", button))

    def tick(self, count, render):
        self.calls.append(("tick", count, render))


class Holder:
    def __init__(self):
        self.pyboy = FakePyBoy()


def test_act_holds_then_releases_and_renders_only_the_last_tick():
    env = Holder()
    act(env, 4)
    assert env.pyboy.calls == [
        ("press", "a"), ("tick", HOLD_TICKS, False), ("release", "a"),
        ("tick", TICKS_PER_ACTION - HOLD_TICKS - 1, False), ("tick", 1, True)]
    assert sum(c[1] for c in env.pyboy.calls if c[0] == "tick") == TICKS_PER_ACTION


def test_pass_presses_nothing():
    env = Holder()
    act(env, PASS)
    assert [c for c in env.pyboy.calls if c[0] != "tick"] == []


def fields(map_id=1, y=2, x=3, levels=(5, 0, 0, 0, 0, 0), badges=0, events=0):
    return {"obs": np.zeros((144, 160, 3), np.uint8), "map_ids": map_id, "xs": x,
            "ys": y, "facings": 0, "in_battles": False, "dialog_opens": False,
            "badges": badges, "party_hps": np.zeros(6, np.int16),
            "party_max_hps": np.zeros(6, np.int16), "party_counts": 1,
            "party_levels": np.array(levels, np.uint8), "event_flag_counts": events}


def episode(n, start_action=0):
    buffer, tracker = EpisodeBuffer(), RewardTracker()
    tracker.start(fields(x=0))
    for t in range(n):
        f = fields(x=t)
        buffer.add(f, (start_action + t) % 8, tracker.step(fields(x=t + 1)),
                   policy_id=1, state_id=3, start=(t == 0))
    return buffer.arrays()


def test_buffer_arrays_match_the_schema():
    arrays = episode(5)
    check(arrays)
    assert arrays["obs"].shape == (5, 144, 160, 3)
    assert arrays["episode_starts"].tolist() == [True, False, False, False, False]
    assert set(POKEDREAMER_KEYS) <= set(SCHEMA)


def test_check_rejects_a_wrong_dtype_and_a_missing_array():
    arrays = episode(3)
    bad = dict(arrays, actions=arrays["actions"].astype(np.int64))
    with pytest.raises(ValueError, match="actions"):
        check(bad)
    with pytest.raises(ValueError, match="missing"):
        check({k: v for k, v in arrays.items() if k != "xs"})


def test_rewards_count_new_tiles_and_gains_only():
    tracker = RewardTracker()
    tracker.start(fields())
    assert tracker.step(fields())["explore"] == 0.0           # same tile
    assert tracker.step(fields(x=4))["explore"] == 1.0         # new tile
    assert tracker.step(fields())["explore"] == 0.0            # been here
    gain = tracker.step(fields(levels=(7, 0, 0, 0, 0, 0)))
    assert gain["level"] == 2.0
    drop = tracker.step(fields(levels=(5, 0, 0, 0, 0, 0)))     # party change
    assert drop["level"] == 0.0
    assert total_reward({"explore": 1, "level": 0, "badge": 1, "event": 0}) == pytest.approx(5.01)


def test_pack_cuts_fixed_chunks_and_keeps_order(tmp_path):
    files = []
    for k, n in enumerate((7, 5, 9)):
        path = tmp_path / f"ep_{k}.npz"
        np.savez_compressed(path, **episode(n, start_action=k))
        files.append(path)
    chunks = pack(files, tmp_path / "out", chunk=8)
    assert [p.name for p in chunks] == ["transitions_0001.npz", "transitions_0002.npz",
                                        "transitions_0003.npz"]
    loaded = [np.load(p) for p in chunks]
    assert [len(c["actions"]) for c in loaded] == [8, 8, 5]
    starts = np.concatenate([c["episode_starts"] for c in loaded])
    assert np.flatnonzero(starts).tolist() == [0, 7, 12]
    actions = np.concatenate([c["actions"] for c in loaded])
    expected = np.concatenate([episode(n, start_action=k)["actions"]
                               for k, n in enumerate((7, 5, 9))])
    assert (actions == expected).all()
    # PokeDreamer's README loader, verbatim in spirit.
    data = np.load(chunks[0])
    assert data["obs"].shape[1:] == (144, 160, 3) and data["actions"].dtype == np.uint8
