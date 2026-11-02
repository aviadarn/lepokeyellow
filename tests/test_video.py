import numpy as np

from poke.video import H, SCALE, W, Snapper, correct, downsample

# PyBoy colour -> what the longplay showed, measured on the bedroom.
MEASURED = [((88, 184, 248), (110, 199, 205)), ((184, 136, 248), (176, 167, 218)),
            ((248, 248, 248), (249, 247, 248)), ((24, 24, 24), (24, 24, 24))]


def test_correction_predicts_the_measured_bedroom_colours():
    for pyboy, seen in MEASURED:
        predicted = correct(np.array([pyboy], np.uint8))[0].astype(int)
        assert np.abs(predicted - np.array(seen)).max() <= 6, (pyboy, predicted, seen)


def test_downsample_takes_the_inner_three_by_three():
    small = np.random.default_rng(0).integers(0, 256, (H, W, 3), dtype=np.uint8)
    big = small.repeat(SCALE, 0).repeat(SCALE, 1)
    big[::SCALE] = 0            # ruin every block's border row and column:
    big[:, ::SCALE] = 0         # the inner 3x3 must not see it
    assert (downsample(big) == small).all()


def test_snapper_inverts_the_correction_through_codec_noise():
    colours = np.array([c for c, _ in MEASURED], np.uint8)
    snapper = Snapper(colours)
    seen = np.array([[s for _, s in MEASURED]], np.uint8)
    noisy = np.clip(seen.astype(int) + [[3, -4, 2]], 0, 255).astype(np.uint8)
    snapped, distance = snapper.snap(noisy)
    assert (snapped[0] == colours).all()
    assert (distance < 30).all()


def test_a_press_under_way_at_a_save_state_is_not_pressed_again():
    from scripts.sync_opening import release_carried

    a, up = 1 << 4, 1 << 0
    masks = np.array([0, a, a, a | up, a, 0, a, a], np.int64)
    out = release_carried(masks, 2)
    # A began at 1, before the state at 2: released until the labels let go;
    # up begins at 3 and the next A at 6 are new presses and stay.
    assert out.tolist() == [0, a, 0, up, 0, 0, a, a]
