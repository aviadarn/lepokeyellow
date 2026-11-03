import torch

from poke.idm import IDM, loss


def test_shapes_and_masked_loss():
    model = IDM(window=8, width=32, layers=1, heads=2)
    frames = torch.randint(0, 42, (2, 9, 144, 160))
    held, mattered, cut, onset = model(frames)
    assert held.shape == mattered.shape == onset.shape == (2, 8, 8)
    assert cut.shape == (2, 8)
    target = torch.randint(0, 2, (2, 8, 8))
    unknown = torch.full((2, 8, 8), 255)
    held_loss, matter_loss = loss(held, mattered, target, unknown)
    assert held_loss > 0 and matter_loss == 0       # no mask computed -> no loss
    known = unknown.clone()
    known[:, 0] = 1
    assert loss(held, mattered, target, known)[1] > 0


def test_first_frame_is_context_only():
    """Changing only the context frame may change outputs, but the window
    length is the frame count minus one."""
    model = IDM(window=8, width=32, layers=1, heads=2).eval()
    frames = torch.randint(0, 42, (1, 6, 144, 160))
    assert model(frames)[0].shape == (1, 5, 8)


def test_spliced_windows_mark_the_join():
    import numpy as np

    from scripts.train_idm import WINDOW, batch

    def episode(value):
        n = 200
        return {"frames": np.full((n, 144, 160), value, np.uint8),
                "buttons": np.zeros(n, np.uint8),
                "mattered": np.full((n, 8), 255, np.uint8)}

    rng = np.random.default_rng(0)
    frames, held, matter, cuts, _ = batch([episode(1), episode(2)], 64, rng, splice=1.0)
    for f, c in zip(frames.numpy(), cuts.numpy()):
        changes = np.flatnonzero(f[1:, 0, 0] != f[:-1, 0, 0])
        joins = np.flatnonzero(c)
        assert len(joins) == 1 and c.shape == (WINDOW,)
        # a join between two different episodes is where the value changes
        if len(changes):
            assert changes.tolist() == joins.tolist()


def test_onset_targets_mask_the_neighbours():
    from poke.idm import onset_targets

    held = torch.zeros(1, 8, 8, dtype=torch.uint8)
    held[0, 3:6, 4] = 1                        # A pressed on frame 3, held to 5
    target, weight = onset_targets(held, torch.zeros(1, 8, dtype=torch.uint8))
    assert target[0, :, 4].tolist() == [0, 0, 0, 1, 0, 0, 0, 0]
    assert weight[0, :, 4].tolist() == [1, 1, 0, 1, 0, 1, 1, 1]
    # held from before the window: no onset
    target, _ = onset_targets(held[:, 4:], held[:, 3])
    assert target[0, 0, 4] == 0


def test_decode_taps_from_onset_peaks():
    import numpy as np

    from scripts.label_video import TAP_FRAMES, decode

    n = 40
    held = np.zeros((n, 8), np.float32)
    held[:, 3] = 0.9                           # right held throughout
    onset = np.zeros((n, 8), np.float32)
    onset[10, 4], onset[11, 4] = 0.8, 0.6      # one A press, NMS keeps frame 10
    masks = decode(held, onset)
    a = (masks >> 4) & 1
    assert a.tolist() == [0] * 10 + [1] * TAP_FRAMES + [0] * (n - 10 - TAP_FRAMES)
    assert ((masks >> 3) & 1).all()


def test_overlapping_directions_keep_the_newest_press():
    import numpy as np

    from scripts.label_video import decode

    n = 30
    held = np.zeros((n, 8), np.float32)
    held[0:15, 0] = 0.9     # up, then the thumb rolls to left at 12
    held[12:30, 2] = 0.9
    held[20:25, 0] = 0.9    # up again while left is still held: up is newest
    masks = decode(held, None)
    up, left = (masks >> 0) & 1, (masks >> 2) & 1
    assert up.tolist() == [1] * 12 + [0] * 8 + [1] * 5 + [0] * 5
    assert left.tolist() == [0] * 12 + [1] * 8 + [0] * 5 + [1] * 5
    assert ((decode(held, None, single_direction=False) >> 0) & 1)[12:15].all()


def test_cuts_need_the_picture_to_jump():
    import numpy as np

    from scripts.label_video import confirm_cuts, remap_share

    rng = np.random.default_rng(0)
    scene = rng.integers(0, 4, (144, 160)).astype(np.uint8)
    faded = (scene + 10).astype(np.uint8)              # same picture, new palette
    menu = faded.copy()
    menu[:, 96:] = 30                                   # a box over 40% of the screen
    other = rng.integers(0, 4, (144, 160)).astype(np.uint8)
    assert remap_share(scene, faded) == 1.0
    assert remap_share(scene, other) < 0.5
    frames = np.stack([scene, faded, menu, other])
    video = {"uframes": frames}
    refs = np.arange(4)
    cuts = np.zeros(4, bool)
    for t in range(1, 4):                               # each pair its own run
        p = np.zeros(4, np.float32)
        p[t] = 0.9
        cuts |= confirm_cuts(p, video, refs)
    assert cuts.tolist() == [False, False, False, True]


def test_context_table_counts_by_situation():
    import numpy as np

    from scripts.eval_idm import context_table

    up, a = 1 << 0, 1 << 4
    truth = np.array([up, 0, a, 0, 0, 0])
    pred = np.array([up, 0, 0, a, 0, 0])
    matter = np.full((6, 8), 255, np.uint8)
    for t in (0, 2, 3, 4, 5):
        matter[t] = 0
    matter[0, 0] = 1                       # up could show on frame 0
    matter[2:5, 4] = 1                     # A could show on frames 2-4
    in_battle = np.array([0, 0, 0, 1, 1, 0], bool)
    dialog = np.array([0, 0, 1, 1, 1, 0], bool)
    assert context_table(truth, pred, matter, in_battle, dialog) == {
        "overworld": {"up": [1, 0, 0, 0]},
        "text box": {"a": [0, 1, 0, 0]},
        "battle": {"a": [0, 0, 1, 1]},
    }


def test_episode_rows_carry_their_batchs_options(tmp_path):
    import json

    from scripts.train_idm import episode_rows

    (tmp_path / "collection.json").write_text(json.dumps({"episodes": [{"file": "a", "state": "s"}]}))
    (tmp_path / "collection_battle.json").write_text(json.dumps(
        {"options": {"encounters": True}, "episodes": [{"file": "b", "state": "s"}]}))
    rows = {r["file"]: r for r in episode_rows(tmp_path)}
    assert rows["b"]["options"] == {"encounters": True}
    assert rows["a"].get("options", {}) == {}


def test_summary_leaves_out_episodes_the_true_buttons_cannot_replay():
    from scripts.eval_idm import summarise

    good = {"truth_replay_exact": True, "idm": {"segments_reproduced": 1.0, "frames": [300],
                                                "same_path": 1.0, "same_end_tile": 1.0}}
    bad = {"truth_replay_exact": False, "idm": {"segments_reproduced": 0.0, "frames": [3],
                                                "same_path": 0.0, "same_end_tile": 0.0}}
    summary = summarise([good, bad], ["idm"])
    assert summary["idm"]["same_path"] == 1.0
    assert summary["excluded_inexact"] == 1 and summary["truth_replay_exact"] is True
