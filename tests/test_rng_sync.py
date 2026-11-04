import io
from pathlib import Path

import numpy as np
import pytest

ROM = Path("pokeyellow.gbc")
STATE = Path("states/cgb/teleport/PalletTown_w0.state")
COLOURS = Path("data/colours_cgb.npy")
pytestmark = pytest.mark.skipif(not (ROM.exists() and STATE.exists() and COLOURS.exists()),
                                reason="needs the ROM, the colour states and the colour table")


def test_find_rng_recovers_the_random_state_that_made_a_recording():
    # Pallet Town's NPCs wander by the random numbers: record 300 frames with
    # them set to (77, 150), then search 64 values for the ones that replay
    # the recording.
    from poke.env import PokeEnv
    from poke.video import Snapper
    from scripts.rng_sync import find_rng, set_rng
    from scripts.sync_opening import Replayer

    env = PokeEnv(str(ROM), str(STATE), color=True)
    env.reset()
    snapper = Snapper(np.load(COLOURS))
    start = io.BytesIO()
    env.pyboy.save_state(start)
    recorder = Replayer(env.pyboy, snapper, np.zeros((1, 144, 160), np.uint8),
                        np.zeros(400, np.int64), 0)
    set_rng(env.pyboy, 77, 150)
    video = []
    for _ in range(300):
        env.pyboy.tick(1, True)
        video.append(recorder.frame())
    video = np.stack(video)

    replayer = Replayer(env.pyboy, snapper, video, np.zeros(400, np.int64), 0)
    snapshot = (start.getvalue(), (0, 0, 0, 0, 0))
    rng = np.random.default_rng(0)
    values = [(77, 150)] + [tuple(int(x) for x in rng.integers(0, 256, 2)) for _ in range(63)]
    found = find_rng(replayer, snapshot, horizon=300, values=values, workers=2)
    assert (77, 150) in found
    assert len(found) < len(values)          # the recording tells the values apart


SEA = Path("states/cgb/teleport/Route20_w0.state")


def test_find_state_ranks_values_by_reach():
    # Route 20's water steps through frames counted in 0xD084: a recording
    # made with it set to 5 is replayed furthest by 5 (0xFFD8 alone only
    # shifts when the next step comes, by under 21 frames: never a loss).
    from poke.env import PokeEnv
    from poke.video import Snapper
    from scripts.rng_sync import find_state
    from scripts.sync_opening import Replayer

    if not SEA.exists():
        pytest.skip("needs the Route 20 colour state")
    env = PokeEnv(str(ROM), str(SEA), color=True)
    env.reset()
    snapper = Snapper(np.load(COLOURS))
    start = io.BytesIO()
    env.pyboy.save_state(start)
    recorder = Replayer(env.pyboy, snapper, np.zeros((1, 144, 160), np.uint8),
                        np.zeros(600, np.int64), 0)
    env.pyboy.memory[0xD084] = 5
    video = []
    for _ in range(400):
        env.pyboy.tick(1, True)
        video.append(recorder.frame())
    replayer = Replayer(env.pyboy, snapper, np.stack(video), np.zeros(600, np.int64), 0)
    ranked = find_state(replayer, (start.getvalue(), (0, 0, 0, 0, 0)), cap=390,
                        registers=((0xD084, range(8)),), workers=2, at=0)
    best = ranked[0][0]
    assert (5,) in [values for reach, values in ranked if reach == best]
    assert ranked[-1][0] < best


def test_find_state_takes_no_chunk_past_its_deadline():
    import time

    from scripts.rng_sync import find_state
    from scripts.sync_opening import Replayer

    class Stub:
        video, masks, slack, patience = np.zeros((10, 144, 160), np.uint8), np.zeros(10), 2, 3

        class snapper:
            colours = np.zeros((4, 3), np.uint8)

    ranked = find_state(Stub(), (b"", (0, 0, 0, 0, 0)), cap=8,
                        registers=((0xFFD3, range(256)),), workers=2, at=0,
                        deadline=time.time() - 1)
    assert ranked == []
