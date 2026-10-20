import numpy as np
import pytest
from PIL import Image

from scripts.record_gif import CAPTION_HEIGHT, caption, caption_font, frame_image


def screen(value=128):
    return np.full((144, 160, 3), value, dtype=np.uint8)


def test_frame_is_the_scaled_screen_plus_a_caption_strip():
    image = frame_image(screen(), 2, ["one", "two"])
    assert image.size == (320, 288 + CAPTION_HEIGHT)
    assert image.mode == "L"


def test_the_screen_is_scaled_without_smoothing():
    """Nearest-neighbour, or a 144x160 Game Boy screen comes out blurred."""
    pixels = screen(0)
    pixels[0, 0] = 255
    image = frame_image(pixels, 4, [""])
    block = np.asarray(image)[:4, :4]
    assert (block == 255).all()


def test_the_caption_is_drawn_over_white_not_over_the_game():
    image = np.asarray(frame_image(screen(0), 2, ["x", "y"]))
    assert (image[:288] == 0).all()          # the screen, untouched
    assert image[288:].max() == 255          # the strip, mostly white
    assert image[288:].min() == 0            # with text on it


def test_caption_reports_the_emulators_truth_and_the_models_guess():
    lines = caption(7, (0, 6, 5), 11.0, "up", (5.1, 5.9))
    assert "Pallet Town" in lines[0] and "(6, 5)" in lines[0]
    assert "11 hops" in lines[0]
    assert "pressed up" in lines[1] and "(5.1, 5.9)" in lines[1]


def test_caption_survives_an_unreadable_position():
    """A -1 read is a failed read; the frame is still worth showing."""
    lines = caption(0, (0, None, None), 99.0, None)
    assert "Pallet Town" in lines[0]
    assert lines[1] == ""


def test_frames_write_a_looping_gif(tmp_path):
    frames = [frame_image(screen(v), 1, [str(v)]) for v in (0, 120, 255)]
    out = tmp_path / "x.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=[800, 100, 800], loop=0, optimize=True)
    with Image.open(out) as reopened:
        assert reopened.n_frames == 3
        assert reopened.info["loop"] == 0


def test_font_scales_with_the_screen():
    small, large = caption_font(1), caption_font(4)
    assert small.getlength("hops") < large.getlength("hops")
