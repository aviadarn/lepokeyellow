"""Evaluation goes through the contract fence, like every reported run."""

import numpy as np
import pytest
import torch

from poke.contract import ContractViolation, PlayTimeEnv
from poke.games import YELLOW
from poke.rl.policy import ActorCritic
from scripts.eval_part2 import NetPolicy, summarise


class FakeEncoder:
    dim, num_frames, out_dim = 4, 2, 8

    def embed(self, frames):
        return np.zeros((len(frames), self.dim), dtype=np.float32)


class FakeEnv:
    game = YELLOW

    def __init__(self):
        self.reads = []

    def screen(self):
        return np.zeros((144, 160, 3), dtype=np.uint8)

    def read_ram(self, addr):
        self.reads.append(addr)
        return 0


def net_policy(**kwargs):
    policy = ActorCritic(feature_dim=4, num_frames=2, hidden=8)
    return NetPolicy(policy, FakeEncoder(), device="cpu", **kwargs)


def test_the_policy_returns_a_legal_action():
    action = net_policy()(PlayTimeEnv(FakeEnv()), [0.0, 0.0, 1.0, 0.0, 0.0])
    assert action in range(7)


def test_the_policy_never_reads_position():
    """The whole point of the fence: this must raise if the policy cheats."""
    inner = FakeEnv()
    guarded = PlayTimeEnv(inner)
    net_policy()(guarded, [0.0, 0.0, 1.0, 0.0, 0.0])
    banned = {YELLOW.ram.cur_map, YELLOW.ram.y_coord, YELLOW.ram.x_coord}
    assert not banned & set(inner.reads)


def test_reaching_for_the_map_through_the_fence_raises():
    guarded = PlayTimeEnv(FakeEnv())
    with pytest.raises(ContractViolation):
        guarded.read_ram(YELLOW.ram.cur_map)


def test_greedy_is_deterministic():
    greedy = net_policy(greedy=True)
    guarded = PlayTimeEnv(FakeEnv())
    first = [greedy(guarded, [0.0] * 5) for _ in range(3)]
    greedy.reset()
    second = [greedy(guarded, [0.0] * 5) for _ in range(3)]
    assert first == second


def test_reset_clears_the_memory_between_runs():
    policy = net_policy()
    guarded = PlayTimeEnv(FakeEnv())
    policy(guarded, [0.0] * 5)
    before = policy.hidden.clone()
    policy.reset()
    assert torch.allclose(policy.hidden, torch.zeros_like(before))


def test_summarise_counts_seeds_that_reached_the_forest():
    runs = [
        {"seed": 0, "objectives": {"get_parcel": 10, "deliver_parcel": 20, "enter_forest": 30},
         "actions": 400, "learned_share": 0.9},
        {"seed": 1, "objectives": {"get_parcel": 11, "deliver_parcel": None, "enter_forest": None},
         "actions": 1461, "learned_share": 0.8},
    ]
    report = summarise(runs)
    assert report["reached"]["enter_forest"] == 1
    assert report["reached"]["get_parcel"] == 2
    assert report["seeds"] == 2
    assert report["median_learned_share"] == pytest.approx(0.85)


def test_summarise_reports_the_design_threshold():
    runs = [{"seed": s, "objectives": {"enter_forest": 1 if s < 5 else None,
                                       "get_parcel": 1, "deliver_parcel": 1},
             "actions": 100, "learned_share": 1.0} for s in range(10)]
    report = summarise(runs)
    assert report["passes_done_threshold"] is True, "5 of 10 is the design's bar"
