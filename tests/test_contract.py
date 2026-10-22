import pytest

from poke.actions import ACTIONS
from poke.contract import ContractViolation, PlayTimeEnv, Run
from poke.games import YELLOW
from poke.objectives import get
from tests.conftest import FakeEnv

A = ACTIONS.index("a")
UP = ACTIONS.index("up")


def base_env(extra=None):
    """RAM addresses are ints, so overrides come in as a dict, not kwargs."""
    return FakeEnv(YELLOW, {
        YELLOW.ram.cur_map: 0, YELLOW.ram.y_coord: 5, YELLOW.ram.x_coord: 5,
        YELLOW.ram.in_battle: 0,
        YELLOW.ram.party_count: 1, YELLOW.ram.event_flags: 0,
        YELLOW.ram.num_bag_items: 0, **(extra or {}),
    })


def test_the_policys_env_refuses_to_report_position():
    guarded = PlayTimeEnv(base_env())
    for addr in (YELLOW.ram.cur_map, YELLOW.ram.y_coord, YELLOW.ram.x_coord):
        with pytest.raises(ContractViolation, match="position"):
            guarded.read_ram(addr)


def test_the_policys_env_refuses_to_roll_back():
    guarded = PlayTimeEnv(base_env())
    with pytest.raises(ContractViolation, match="rollback"):
        guarded.snapshot()
    with pytest.raises(ContractViolation, match="rollback"):
        guarded.restore(b"")


def test_flags_and_the_bag_still_read_fine():
    guarded = PlayTimeEnv(base_env({YELLOW.ram.event_flags: 0b100000}))
    assert guarded.read_ram(YELLOW.ram.event_flags) == 0b100000


def test_the_screen_still_comes_through():
    """Pixels are the whole point: the guard blocks position, not vision."""
    assert PlayTimeEnv(base_env()).screen().shape == (144, 160, 3)


def test_learned_share_counts_only_the_policys_actions():
    run = Run(base_env())
    run.record(UP, "policy")
    run.record(A, "text")
    run.record(A, "battle")
    run.record(UP, "policy")
    assert run.by_controller == {"policy": 2, "text": 1, "battle": 1}
    assert run.learned_share == 0.5


def test_no_control_turns_count_as_neither():
    """Walking cutscenes and the swallowed input after a warp are nobody's."""
    run = Run(base_env())
    run.record(UP, "policy")
    run.record(ACTIONS.index("noop"), "no_control")
    assert run.by_controller["no_control"] == 1
    assert run.learned_share == 1.0


def test_two_tile_step_counts_as_one_action_and_two_new_tiles():
    """One 60-frame action sometimes walks two tiles, so consecutive positions
    are not always adjacent. Tile counting must not assume they are."""
    env = base_env()
    run = Run(env)
    run.record(UP, "policy")
    env.write(YELLOW.ram.y_coord, 3)
    run.record(UP, "policy")
    assert len(run.log) == 2
    assert run.tiles_seen == {(0, 5, 5), (0, 3, 5)}


def test_blackout_warp_is_logged_and_does_not_reset_the_objective():
    """Blacking out is not special-cased: the player reappears at home and the
    run carries on paying for it in actions."""
    env = base_env()
    run = Run(env)
    run.record(UP, "policy")
    env.write(YELLOW.ram.cur_map, 37)   # warped home
    env.write(YELLOW.ram.y_coord, 1)
    env.write(YELLOW.ram.x_coord, 2)
    run.record(UP, "policy")
    assert run.log[-1]["map"] == 37
    assert (37, 1, 2) in run.tiles_seen
    assert run.objective_key == get("trigger_oak").key


def test_new_tiles_reset_when_the_objective_changes():
    """The visited set is per objective, which is what makes the backtrack to
    Oak pay a second time."""
    env = base_env()
    run = Run(env)
    run.record(UP, "policy")
    # OAK_APPEARED_IN_PALLET is bit 39: byte 4, bit 7.
    env.write(YELLOW.ram.event_flags + 39 // 8, 1 << (39 % 8))
    run.record(UP, "policy")
    assert run.new_tiles_this_objective == 1


def test_the_guard_is_deny_by_default_not_two_blocked_names():
    """Every route to position the review found, closed.

    __getattr__ only fires for attributes the object does not have, and
    enumerating two forbidden names leaves the rest of PokeEnv forwarded, so
    the fence was a suggestion: `_env`, `pyboy`, `read_ram_range` and
    `screen_of_state` all reached the map id or rolled the emulator back.
    """
    guarded = PlayTimeEnv(base_env())
    for attribute in ("_env", "pyboy", "read_ram_range", "screen_of_state",
                      "snapshot", "restore", "init_state"):
        with pytest.raises(ContractViolation):
            getattr(guarded, attribute)


def test_hasattr_answers_false_instead_of_exploding():
    """hasattr only swallows AttributeError, so a duck-typing caller -- copy,
    a gym wrapper, a logger -- used to take a ContractViolation in the face."""
    guarded = PlayTimeEnv(base_env())
    assert not hasattr(guarded, "snapshot")
    assert not hasattr(guarded, "pyboy")
    assert hasattr(guarded, "screen")


def test_the_allowed_surface_is_what_a_policy_needs_and_no_more():
    guarded = PlayTimeEnv(base_env())
    assert guarded.screen().shape == (144, 160, 3)
    assert guarded.read_ram(YELLOW.ram.event_flags) == 0
    assert guarded.game is not None


def test_neighbouring_addresses_are_blocked_too():
    """The map-header pointer at cur_map+1 is a perfect map-id proxy, and the
    block coords sit just past x. Blocking three exact addresses was a value
    check an adversary walks around by reading the byte next door."""
    guarded = PlayTimeEnv(base_env())
    for addr in (YELLOW.ram.cur_map + 1, YELLOW.ram.cur_map + 2,
                 YELLOW.ram.y_coord + 2, YELLOW.ram.x_coord + 2):
        with pytest.raises(ContractViolation, match="position"):
            guarded.read_ram(addr)
