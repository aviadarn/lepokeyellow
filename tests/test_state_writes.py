import numpy as np

from poke.video import Snapper
from scripts.sync_opening import Replayer, rows_from_writes, writes_from_rows

COLOURS = np.array([(0, 0, 0), (8, 8, 8), (248, 248, 0), (248, 248, 248)], np.uint8)


class _Memory(dict):
    def __init__(self, ticks):
        super().__init__()
        self.ticks, self.log = ticks, []

    def __setitem__(self, address, value):
        self.log.append((self.ticks[0], address, value))
        super().__setitem__(address, value)


class _Screen:
    ndarray = np.full((144, 160, 4), 248, np.uint8)


class _FakePyBoy:
    def __init__(self):
        self.ticks = [0]
        self.memory = _Memory(self.ticks)
        self.memory.update({0xFF43: 0, 0xFF42: 0})
        self.screen = _Screen()

    def tick(self, *args):
        self.ticks[0] += 1

    def button_press(self, name):
        pass

    def button_release(self, name):
        pass

    def save_state(self, buffer):
        buffer.write(b"state")

    def load_state(self, buffer):
        pass


def _replayer(writes):
    pyboy = _FakePyBoy()
    video = np.full((20, 144, 160), 3, np.uint8)
    replayer = Replayer(pyboy, Snapper(COLOURS), video, np.zeros(20, np.int64), 0)
    replayer.writes = writes
    return replayer, pyboy


def test_writes_round_trip_through_rows():
    writes = {13008: ((0xFFD8, 0), (0xD084, 1)), 900: ((0xFFD3, 77),)}
    rows = rows_from_writes(writes)
    assert rows.dtype == np.int32 and rows.shape == (3, 3)
    assert rows[:, 0].tolist() == [900, 13008, 13008]
    assert writes_from_rows(rows) == writes
    assert writes_from_rows(np.zeros((0, 3), np.int32)) == {}


def test_a_write_applies_once_on_its_frame():
    replayer, pyboy = _replayer({3: ((0xFFD8, 7),)})
    for _ in range(6):
        replayer.step()
    assert pyboy.memory.log == [(3, 0xFFD8, 7)]     # before the tick that draws frame 3


def test_a_write_before_the_snapshot_is_not_replayed():
    replayer, pyboy = _replayer({3: ((0xFFD8, 7),)})
    for _ in range(5):
        replayer.step()
    snapshot = replayer.save()
    pyboy.memory.log.clear()
    replayer.load(snapshot)
    for _ in range(4):
        replayer.step()
    assert pyboy.memory.log == []


def test_a_video_window_reads_only_its_frames_by_absolute_index():
    import pytest

    from scripts.sync_opening import VideoWindow, video_frames

    refs = np.repeat(np.arange(50), 2).astype(np.int32)          # 100 frames, 50 unique
    uframes = np.arange(50, dtype=np.uint8)[:, None, None] * np.ones((1, 144, 160), np.uint8)
    h5 = {"frame_ref": refs, "uframes": uframes}
    full = video_frames(h5, 0, 100)
    window = VideoWindow(h5, 60, 30)
    assert len(window) == 90
    assert np.array_equal(window[75], full[75])
    assert np.array_equal(window[62:70], full[62:70])
    with pytest.raises(IndexError):
        window[59]
