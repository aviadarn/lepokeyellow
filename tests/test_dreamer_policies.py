import numpy as np

from poke.dreamer_policies import (A, DIRECTIONS, DOWN, LEFT, RIGHT, UP,
                                   Frontier, StickyRandom, make)


def fields(y=5, x=5, battle=False, dialog=False):
    return {"map_ids": 1, "ys": y, "xs": x, "in_battles": battle, "dialog_opens": dialog}


def test_sticky_random_is_seeded_and_in_range():
    p, q = StickyRandom(np.random.default_rng(3)), StickyRandom(np.random.default_rng(3))
    seq_p = [p(fields()) for _ in range(200)]
    seq_q = [q(fields()) for _ in range(200)]
    assert seq_p == seq_q
    assert set(seq_p) <= set(range(8))


def test_frontier_learns_a_wall_and_stops_walking_into_it():
    policy = Frontier(np.random.default_rng(0), epsilon=0.0, momentum=0.0)
    policy.last = RIGHT
    policy.previous = (1, 5, 5)
    policy(fields())               # RIGHT left us on (5, 5): a wall
    assert ((1, 5, 5), RIGHT) in policy.blocked
    for _ in range(50):
        assert policy(fields()) != RIGHT or policy.blocked == set()
        policy.previous = None     # pretend each choice is evaluated fresh


def test_frontier_prefers_the_unvisited_neighbour():
    policy = Frontier(np.random.default_rng(0), epsilon=0.0, momentum=0.0)
    for direction in DIRECTIONS:
        if direction != DOWN:
            policy.visits[policy._step((1, 5, 5), direction)] = 9
    assert policy(fields()) == DOWN


def test_frontier_mostly_presses_a_in_text():
    policy = Frontier(np.random.default_rng(0))
    presses = [policy(fields(dialog=True)) for _ in range(500)]
    assert presses.count(A) / len(presses) > 0.5


def test_make_dispatches_on_policy_id():
    assert isinstance(make(0, np.random.default_rng(0)), StickyRandom)
    assert isinstance(make(1, np.random.default_rng(0)), Frontier)


def test_frontier_can_steer_a_battle_menu_back_to_fight():
    """FIGHT is top left. A menu policy without LEFT gets stuck on RUN."""
    policy = Frontier(np.random.default_rng(0))
    presses = [policy(fields(battle=True)) for _ in range(2000)]
    assert presses.count(LEFT) > 0 and presses.count(UP) > 0
    assert presses.count(LEFT) + presses.count(UP) > presses.count(DOWN) + presses.count(RIGHT)
