import json

import pytest

from poke.actions import ACTIONS
from poke.routes import SCHEMA, Route, state_digest


def write_state(tmp_path, payload=b"state-bytes"):
    path = tmp_path / "yellow_lab.state"
    path.write_bytes(payload)
    return path


def make_route(state_path, **overrides):
    fields = dict(
        schema=SCHEMA,
        game="yellow",
        task="yellow_pikachu",
        state_sha1=state_digest(state_path),
        actions=["up", "up", "a", "a"],
        reached_goal=True,
        goal_step=4,
    )
    fields.update(overrides)
    return Route(**fields)


def test_roundtrip(tmp_path):
    state = write_state(tmp_path)
    route = make_route(state)
    path = tmp_path / "routes" / "r.json"
    route.save(path)
    assert Route.load(path) == route
    assert route.indices == [ACTIONS.index(n) for n in route.actions]


def test_rejects_a_route_from_a_different_save_state(tmp_path):
    """Replaying a route against the wrong state yields episodes labelled by a
    predicate that never fired -- a silently wrong dataset."""
    state = write_state(tmp_path)
    route = make_route(state)
    state.write_bytes(b"different-bytes")
    with pytest.raises(ValueError, match="different save state"):
        route.check("yellow", "yellow_pikachu", state)


def test_rejects_a_route_from_another_game_or_task(tmp_path):
    state = write_state(tmp_path)
    route = make_route(state)
    with pytest.raises(ValueError, match="recorded for 'yellow'"):
        route.check("red", "yellow_pikachu", state)
    with pytest.raises(ValueError, match="task 'yellow_pikachu'"):
        route.check("yellow", "leave_room", state)


def test_rejects_a_route_that_never_reached_the_goal(tmp_path):
    state = write_state(tmp_path)
    route = make_route(state, reached_goal=False, goal_step=-1)
    with pytest.raises(ValueError, match="never reaching the goal"):
        route.check("yellow", "yellow_pikachu", state)


def test_rejects_unknown_action_names(tmp_path):
    state = write_state(tmp_path)
    path = tmp_path / "bad.json"
    data = make_route(state).__dict__ | {"actions": ["up", "start"]}
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="unknown action names"):
        Route.load(path)


def test_rejects_a_future_schema(tmp_path):
    state = write_state(tmp_path)
    path = tmp_path / "bad.json"
    data = make_route(state).__dict__ | {"schema": SCHEMA + 1}
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="route schema"):
        Route.load(path)
