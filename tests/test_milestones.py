import pytest

from poke.games import YELLOW
from poke.maps import OAKS_LAB, PALLET_TOWN, REDS_HOUSE_2F
from poke.milestones import CHAIN, chain_from, get
from poke.ram import NO_BATTLE, TRAINER_BATTLE
from tests.conftest import FakeEnv


def env_at(map_id=OAKS_LAB, party=0, battle=NO_BATTLE):
    return FakeEnv(YELLOW, {
        YELLOW.ram.cur_map: map_id,
        YELLOW.ram.party_count: party,
        YELLOW.ram.in_battle: battle,
    })


def test_chain_order_is_the_scoped_target():
    assert [m.key for m in CHAIN] == [
        "pikachu", "rival_beaten", "left_lab", "left_pallet"
    ]


def test_chain_from_skips_already_cleared_milestones():
    assert [m.key for m in chain_from("left_lab")] == ["left_lab", "left_pallet"]
    assert chain_from() == CHAIN
    with pytest.raises(ValueError, match="unknown milestone"):
        chain_from("gym_badge")


def test_pikachu_milestone():
    assert not get("pikachu").reached(env_at(party=0), {})
    assert get("pikachu").reached(env_at(party=1), {})


def test_rival_beaten_needs_the_battle_to_have_happened():
    """"Not in a battle" is also true BEFORE the fight, so the predicate has to
    consult the history or it fires immediately."""
    milestone = get("rival_beaten")
    assert not milestone.reached(env_at(party=1), {})
    assert not milestone.reached(env_at(party=1, battle=TRAINER_BATTLE),
                                 {"saw_trainer_battle": True})
    assert milestone.reached(env_at(party=1), {"saw_trainer_battle": True})


def test_rival_beaten_requires_a_surviving_party():
    milestone = get("rival_beaten")
    assert not milestone.reached(env_at(party=0), {"saw_trainer_battle": True})


def test_left_lab_requires_the_pokemon_too():
    """Walking out empty-handed is not progress along this chain."""
    milestone = get("left_lab")
    assert not milestone.reached(env_at(map_id=PALLET_TOWN, party=0), {})
    assert not milestone.reached(env_at(map_id=OAKS_LAB, party=1), {})
    assert milestone.reached(env_at(map_id=PALLET_TOWN, party=1), {})


def test_left_pallet_excludes_every_map_inside_pallet():
    milestone = get("left_pallet")
    for map_id in (PALLET_TOWN, REDS_HOUSE_2F, OAKS_LAB):
        assert not milestone.reached(env_at(map_id=map_id, party=1), {})
    assert milestone.reached(env_at(map_id=12, party=1), {})


def test_unreadable_map_is_not_progress():
    env = FakeEnv(YELLOW, {YELLOW.ram.party_count: 1})
    assert not get("left_lab").reached(env, {})
    assert not get("left_pallet").reached(env, {})


def test_milestones_without_a_goal_image_are_flagged_unplannable():
    """The battle is cleared by a scripted policy and leaving Pallet has no
    bootstrap state yet, so CEM has nothing to steer towards for either."""
    plannable = {m.key: m.is_plannable() for m in CHAIN}
    assert plannable == {
        "pikachu": True, "rival_beaten": False,
        "left_lab": True, "left_pallet": False,
    }


def test_frame_to_tensor_scaling_matches_the_dataset():
    """play.py must normalise a live screen exactly as the dataset loader does,
    or the encoder sees a different distribution at play time than at train
    time."""
    import numpy as np
    import torch as t

    from scripts.play import to_tensor

    frame = np.zeros((144, 160, 3), dtype=np.uint8)
    frame[:, :, 0] = 255  # channel 0 is luminance in DMG mode
    tensor = to_tensor(frame, t.device("cpu"))
    assert tensor.shape == (1, 1, 144, 160)
    assert abs(float(tensor.max()) - (255 / 127.5 - 1.0)) < 1e-6

    frame[:, :, 0] = 0
    assert abs(float(to_tensor(frame, t.device("cpu")).min()) + 1.0) < 1e-6


def test_every_plannable_milestone_has_its_goal_state_on_disk():
    """A milestone pointing at a state file nobody bootstrapped would make
    play.py fail halfway through a run instead of at startup."""
    from pathlib import Path

    for milestone in CHAIN:
        if milestone.is_plannable():
            assert Path(milestone.goal_state).exists(), milestone.goal_state
