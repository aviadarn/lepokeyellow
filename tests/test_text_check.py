import numpy as np

from scripts.sync_opening import (SYNC_SHARE, TEXT_LIMIT, _patch_sums, in_sync, match,
                                  text_classes, text_mismatch, text_mismatch_near)

# Snapper order (sorted): black, near-black ink, yellow art, white paper.
COLOURS = np.array([(0, 0, 0), (8, 8, 8), (248, 248, 0), (248, 248, 248)], np.uint8)
INK, INK2, ART, PAPER = 0, 1, 2, 3
K = ["X...X..", "X..X...", "X.X....", "XX.....", "X.X....", "X..X...", "X...X.."]
R = ["XXXX...", "X...X..", "X...X..", "XXXX...", "X.X....", "X..X...", "X...X.."]


def screen(glyph=None, colour=INK, at=(120, 40)):
    frame = np.full((144, 160), PAPER, np.uint8)
    if glyph:
        y, x = at
        for dy, row in enumerate(glyph):
            for dx, c in enumerate(row):
                if c == "X":
                    frame[y + dy, x + dx] = colour
    return frame


def test_paper_ink_and_everything_else():
    classes = text_classes(COLOURS)
    assert classes[[INK, INK2, ART, PAPER]].tolist() == [1, 1, 2, 0]
    assert classes[255] == 2          # a pixel the snap could not place is not text


def test_a_wrong_letter_is_a_mismatch_the_pixel_share_misses():
    classes = text_classes(COLOURS)
    video, emulator = screen(R), screen(K)
    assert match(emulator, video) >= SYNC_SHARE
    assert text_mismatch(emulator, video, classes) >= TEXT_LIMIT
    assert not in_sync(match(emulator, video), text_mismatch(emulator, video, classes))


def test_the_same_letter_in_another_ink_shade_matches():
    classes = text_classes(COLOURS)
    assert text_mismatch(screen(K, INK), screen(K, INK2), classes) == 0


def test_coloured_art_is_not_judged_as_text():
    # Title Pikachu's face and wandering sprites differ in colours that are
    # not paper and ink: the pixel share judges them, not the text check.
    classes = text_classes(COLOURS)
    assert text_mismatch(screen(K, ART), screen(R, ART), classes) == 0


def test_a_letter_printed_a_few_frames_early_is_timing_not_content():
    classes = text_classes(COLOURS)
    video = np.stack([screen(K) if t >= 10 else screen() for t in range(40)])
    emulator = screen(K)
    assert text_mismatch(emulator, video[5], classes) >= TEXT_LIMIT
    assert text_mismatch_near(emulator, video, 5, classes, slack=12) == 0
    assert text_mismatch_near(emulator, video, 5, classes, slack=3) >= TEXT_LIMIT
    # A wrong letter matches no frame, however far the search looks.
    assert text_mismatch_near(screen(R), video, 20, classes, slack=12) >= TEXT_LIMIT


class _ScriptedReplayer:
    """Replays nothing: step() reports the text mismatch scripted per frame."""

    def __init__(self, text):
        self.text, self.e, self.offset, self.masks = text, 0, 0, None

    def save(self):
        return 0

    def load(self, snapshot):
        self.e = snapshot

    def step(self):
        v = self.e
        self.e += 1
        return v, 1.0, int(self.text[v]) if v < len(self.text) else 0


def test_repair_search_loses_sync_on_wrong_text():
    from scripts.repair_labels import RUN, Search

    text = np.zeros(400, np.int64)
    text[50:50 + RUN] = TEXT_LIMIT           # a wrong name, on screen RUN frames
    text[200:200 + RUN - 1] = TEXT_LIMIT     # shorter than RUN: not a loss
    search = Search(_ScriptedReplayer(text), 0, 400)
    assert search.play(0, None, 400) == 50
    text[50:50 + RUN] = 0
    assert search.play(0, None, 400) == 400


def _reference_text_mismatch(a, b, classes):
    """text_mismatch as first written: the patch maximum, always computed."""
    ca, cb = classes[a], classes[b]
    judged = _patch_sums((ca == 2) | (cb == 2)) == 0
    if not judged.any():
        return 0
    return int(_patch_sums(ca != cb)[judged].max())


def _random_screen(rng):
    frame = screen()
    for _ in range(rng.integers(0, 4)):
        glyph = (K, R)[rng.integers(2)]
        colour = (INK, INK2, ART, 255)[rng.integers(4)]
        at = (int(rng.integers(0, 136)), int(rng.integers(0, 152)))
        y, x = at
        for dy, row in enumerate(glyph):
            for dx, c in enumerate(row):
                if c == "X" and y + dy < 144 and x + dx < 160:
                    frame[y + dy, x + dx] = colour
    return frame


def test_fast_text_check_agrees_with_the_full_one():
    classes = text_classes(COLOURS)
    rng = np.random.default_rng(0)
    for _ in range(200):
        a, b = _random_screen(rng), _random_screen(rng)
        assert (text_mismatch(a, b, classes) >= TEXT_LIMIT) == \
            (_reference_text_mismatch(a, b, classes) >= TEXT_LIMIT)


class _Watched:
    """A video that records which frames are read."""

    def __init__(self, frames):
        self.frames, self.read = frames, set()

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, u):
        self.read.add(int(u))
        return self.frames[u]


def test_near_stops_once_every_patch_has_a_match():
    classes = text_classes(COLOURS)
    frames = np.stack([screen() for _ in range(30)])
    frames[16] = screen(K)
    video = _Watched(frames)
    assert text_mismatch_near(screen(K), video, 15, classes) == 0
    assert video.read <= {14, 15, 16}          # v, then v-1, v+1: no further


def test_letters_shown_in_other_batches_match_patch_by_patch():
    # The game copies the screen a third at a time, so the capture emulator
    # can show "K", then "K R K" where PyBoy shows "K R": no single video
    # frame equals it, but every letter matches some frame.
    classes = text_classes(COLOURS)
    one = screen(K, at=(120, 40))
    three = screen(K, at=(120, 40))
    for x, g in ((48, R), (56, K)):
        three = np.minimum(three, screen(g, at=(120, x)))
    video = np.stack([screen()] * 5 + [one] * 2 + [three] * 10)
    emulator = np.minimum(screen(K, at=(120, 40)), screen(R, at=(120, 48)))
    assert min(text_mismatch(emulator, f, classes) for f in video) >= TEXT_LIMIT
    assert text_mismatch_near(emulator, video, 6, classes) < TEXT_LIMIT


def test_a_wrong_letter_fails_patch_by_patch_too():
    classes = text_classes(COLOURS)
    video = np.stack([screen(K, at=(120, 40))] * 30)
    assert text_mismatch_near(screen(R, at=(120, 40)), video, 15, classes) >= TEXT_LIMIT


class _FakeScreen:
    def __init__(self, rgb):
        self.ndarray = np.concatenate([rgb, np.full(rgb.shape[:2] + (1,), 255, np.uint8)], -1)


class _FakePyBoy:
    def __init__(self, rgb):
        self.screen = _FakeScreen(rgb)

    def tick(self, *args):
        pass

    def button_press(self, name):
        pass

    def button_release(self, name):
        pass


def test_text_is_not_judged_when_the_share_failed():
    from poke.video import Snapper
    from scripts.sync_opening import Replayer

    snapper = Snapper(COLOURS)
    left_half = np.full((144, 160, 3), 248, np.uint8)
    left_half[:, :80] = 0                    # half black: share 0.5 against white
    video = np.stack([screen()] * 10)
    replayer = Replayer(_FakePyBoy(left_half), snapper, video, np.zeros(10, np.int64), 0)
    replayer.e = 4
    v, share, text = replayer.step()
    assert share < SYNC_SHARE and text is None
    assert not in_sync(share, text) and not in_sync(0.99, None)
