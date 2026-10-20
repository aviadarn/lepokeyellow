import torch as t

from poke.coord_head import CoordHead, coord_loss


def batch(b=4, time=4, dim=192, seed=0):
    t.manual_seed(seed)
    predictions = t.randn(b, time - 1, dim)
    targets = t.randn(b, time - 1, dim)
    location = t.stack([
        t.zeros(b, time, dtype=t.long),
        t.randint(0, 10, (b, time)),
        t.randint(0, 10, (b, time)),
    ], dim=-1)
    return predictions, targets, location


def test_loss_is_differentiable_through_the_prediction():
    """The whole point: the gradient has to reach the predictor, or the
    predicted embedding is never asked to keep the player's tile legible."""
    predictions, targets, location = batch()
    predictions.requires_grad_(True)
    head = CoordHead()
    loss, _ = coord_loss(head, predictions, targets, location)
    loss.backward()
    assert predictions.grad is not None and predictions.grad.abs().sum() > 0


def test_unreadable_positions_are_masked_out():
    """A -1 read is a failed read, not a position at (-1, -1)."""
    predictions, targets, location = batch()
    head = CoordHead()
    full, _ = coord_loss(head, predictions, targets, location)

    masked = location.clone()
    masked[:, 1:, 0] = -1
    none_left, tiles = coord_loss(head, predictions, targets, masked)
    assert float(none_left.detach()) == 0.0 and float(tiles) == 0.0
    assert float(full.detach()) > 0.0


def test_reported_tiles_are_the_predicted_error_not_the_loss():
    predictions, targets, location = batch()
    head = CoordHead()
    _, tiles = coord_loss(head, predictions, targets, location)
    with t.no_grad():
        wanted = location[:, 1:, 1:].float()
        expected = (head(predictions) - wanted).abs().sum(dim=-1).mean()
    assert abs(float(tiles) - float(expected)) < 1e-5


def test_head_reads_two_numbers_from_any_leading_shape():
    head = CoordHead(dim=8, hidden=16)
    assert head(t.randn(3, 8)).shape == (3, 2)
    assert head(t.randn(3, 5, 8)).shape == (3, 5, 2)
