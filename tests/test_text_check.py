import numpy as np

from scripts.sync_opening import (SYNC_SHARE, TEXT_LIMIT, in_sync, match, text_classes,
                                  text_mismatch, text_mismatch_near)

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
