import pytest

from scripts.play import continuation


def test_continuation_tiles_become_waypoints_with_the_press_they_need():
    """The recorded route ends on the frame Pikachu is handed over, so three of
    the four milestones are past the end of the data. A door is a step and the
    top edge of a map is a step, so each continuation tile carries the
    direction that goes through it."""
    points = continuation("40,11,5,down;0,0,10,up", offset=38)
    assert [(p.map_id, p.y, p.x) for p in points] == [(40, 11, 5), (0, 0, 10)]
    assert [p.facing for p in points] == ["down", "up"]
    assert all(p.scripted for p in points)
    assert [p.route_step for p in points] == [38, 39]


def test_a_continuation_tile_without_a_press_is_an_ordinary_waypoint():
    point, = continuation("0,5,5", offset=0)
    assert not point.scripted and point.facing is None


def test_an_empty_continuation_is_no_waypoints():
    assert continuation("", offset=0) == []


def test_a_malformed_continuation_fails_loudly():
    with pytest.raises(SystemExit, match="map,y,x"):
        continuation("40,11", offset=0)
    with pytest.raises(SystemExit, match="not a direction"):
        continuation("40,11,5,sideways", offset=0)
