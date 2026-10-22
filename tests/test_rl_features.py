"""The frozen encoder, and the rolling window of embeddings fed to the policy."""

import numpy as np
import pytest
import torch

from poke.jepa import JEPA
from poke.rl.features import EmbeddingStack, FrozenEncoder, checkpoint_geometry


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    """A real JEPA checkpoint in the shape scripts/train.py writes."""
    path = tmp_path_factory.mktemp("ckpt") / "best.pt"
    model = JEPA(num_frames=4, dim=192)
    torch.save({"jepa": model.state_dict(), "config": {"num_frames": 4}}, path)
    return path


def test_geometry_is_read_from_the_weights_and_the_config():
    blob = {"jepa": {"encoder.cls": torch.zeros(1, 1, 128)}, "config": {"num_frames": 3}}
    assert checkpoint_geometry(blob) == (128, 3)


def test_checkpoint_geometry_mismatch_raises(tmp_path):
    """Review focus 2: a checkpoint the policy was not built for."""
    path = tmp_path / "wrong.pt"
    torch.save({"jepa": {"encoder.cls": torch.zeros(1, 1, 128)},
                "config": {"num_frames": 3}}, path)
    with pytest.raises(ValueError) as caught:
        FrozenEncoder(path, device="cpu", expect_frames=4)
    message = str(caught.value)
    assert "num_frames" in message and "3" in message and "4" in message


def test_embed_returns_one_vector_per_frame(checkpoint):
    encoder = FrozenEncoder(checkpoint, device="cpu")
    frames = np.zeros((2, 144, 160), dtype=np.uint8)
    out = encoder.embed(frames)
    assert out.shape == (2, 192)
    assert out.dtype == np.float32


def test_embed_takes_no_gradients(checkpoint):
    encoder = FrozenEncoder(checkpoint, device="cpu")
    assert all(not p.requires_grad for p in encoder.model.parameters())


def test_out_dim_is_the_whole_window(checkpoint):
    encoder = FrozenEncoder(checkpoint, device="cpu")
    assert encoder.out_dim == 4 * 192


def test_the_stack_starts_full():
    stack = EmbeddingStack(num_frames=4, dim=3)
    stack.reset(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert stack.vector().tolist() == [1, 2, 3] * 4


def test_the_stack_rolls_oldest_out():
    stack = EmbeddingStack(num_frames=2, dim=1)
    stack.reset(np.array([1.0], dtype=np.float32))
    stack.push(np.array([2.0], dtype=np.float32))
    stack.push(np.array([3.0], dtype=np.float32))
    assert stack.vector().tolist() == [2.0, 3.0], "oldest first, newest last"


def test_pushing_before_reset_is_an_error():
    with pytest.raises(RuntimeError):
        EmbeddingStack(num_frames=2, dim=1).push(np.zeros(1, dtype=np.float32))


def test_half_precision_embeds_to_the_same_shape_and_dtype(checkpoint):
    """The encoder is frozen, so fp16 inference costs nothing but precision,
    and the per-call MPS latency is what bounds the training loop."""
    encoder = FrozenEncoder(checkpoint, device="cpu", half=True)
    out = encoder.embed(np.zeros((2, 144, 160), dtype=np.uint8))
    assert out.shape == (2, 192)
    assert out.dtype == np.float32, "the policy always sees float32"


def test_half_precision_stays_close_to_full(checkpoint):
    frames = np.full((2, 144, 160), 128, dtype=np.uint8)
    full = FrozenEncoder(checkpoint, device="cpu").embed(frames)
    half = FrozenEncoder(checkpoint, device="cpu", half=True).embed(frames)
    assert np.abs(full - half).max() < 0.1
