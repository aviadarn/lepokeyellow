"""The driver: rollout collection, the kill rule, and surviving a dead worker."""

import numpy as np
import torch

from scripts.train_rl import TrainConfig, Trainer, should_stop


class FakeHandle:
    """A worker that answers in-process, and can be told to die or to finish."""

    def __init__(self, map_id=12, dies_at=None, done_at=None):
        self.map_id = map_id
        self.dies_at = dies_at
        self.done_at = done_at
        self.calls = 0
        self.restarts = 0
        self.sent = []

    def send(self, message):
        self.sent.append(message)

    def recv(self):
        if self.sent and self.sent[-1][0] == "snapshot":
            return b"blob-%d" % self.map_id
        self.calls += 1
        if self.dies_at is not None and self.calls >= self.dies_at:
            raise EOFError("worker gone")
        return {
            "frame": np.zeros((144, 160), dtype=np.uint8),
            "objective": [0.0, 0.0, 1.0, 0.0, 0.0],
            "reward": 0.0,
            "done": self.done_at is not None and self.calls == self.done_at,
            "info": {"map": self.map_id, "y": 1, "x": 1, "completed": [],
                     "controllers": {"policy": 1}, "taken": 1, "can_snapshot": True},
        }

    def restart(self):
        self.restarts += 1
        self.calls = 0


class OrderedHandle(FakeHandle):
    """Records sends and receives into a shared log, to prove concurrency."""

    def __init__(self, name, log, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.log = log

    def send(self, message):
        super().send(message)
        self.log.append(("send", self.name, message[0]))

    def recv(self):
        reply = super().recv()
        self.log.append(("recv", self.name))
        return reply


class FakeEncoder:
    dim = 4
    num_frames = 2
    out_dim = 8

    def embed(self, frames):
        return np.zeros((len(frames), self.dim), dtype=np.float32)


def trainer(handles, **overrides):
    from poke.rl.archive import Archive
    from poke.rl.policy import ActorCritic
    config = TrainConfig(**{"rollout_steps": 4, **overrides})
    encoder = FakeEncoder()
    policy = ActorCritic(feature_dim=encoder.dim, num_frames=encoder.num_frames, hidden=8)
    return Trainer(handles=handles, policy=policy, encoder=encoder,
                   archive=Archive(), bonus=None, config=config, device="cpu")


def test_collect_fills_the_buffer_and_steps_every_worker():
    handles = [FakeHandle(), FakeHandle()]
    driver = trainer(handles)
    buffer = driver.new_buffer()
    driver.collect(buffer)
    assert buffer.obs.shape[0] == 4
    assert sum(1 for m in handles[0].sent if m[0] == "step") == 4


def test_a_dead_worker_is_restarted_and_its_partial_rollout_dropped():
    """Review focus 5."""
    alive, doomed = FakeHandle(), FakeHandle(dies_at=3)
    driver = trainer([alive, doomed])
    buffer = driver.new_buffer()
    report = driver.collect(buffer)
    assert doomed.restarts == 1, "restarted once, not once per remaining step"
    assert report["dropped_workers"] == [1]
    assert report["usable_envs"] == [0], "the dead worker's column is not trained on"


def test_a_dropped_worker_is_not_polled_again_in_the_same_rollout():
    alive, doomed = FakeHandle(), FakeHandle(dies_at=2)
    driver = trainer([alive, doomed])
    driver.collect(driver.new_buffer())
    assert doomed.calls == 0, "left alone after its restart"


def test_the_first_collect_starts_from_the_archive_and_the_second_does_not():
    """A 512-action episode cannot survive a reset every 128-step rollout."""
    handles = [FakeHandle()]
    driver = trainer(handles)
    driver.collect(driver.new_buffer())
    resets_after_first = sum(1 for m in handles[0].sent if m[0] == "reset")
    driver.collect(driver.new_buffer())
    resets_after_second = sum(1 for m in handles[0].sent if m[0] == "reset")
    assert resets_after_first == 1
    assert resets_after_second == 1, "the second rollout continues the episode"


def test_a_finished_episode_is_restarted_and_its_memory_cleared():
    handles = [FakeHandle(done_at=2)]
    driver = trainer(handles)
    driver.hidden = torch.ones_like(driver.hidden)
    driver.collect(driver.new_buffer())
    assert sum(1 for m in handles[0].sent if m[0] == "reset") == 2, "start, then after done"


def test_the_kill_rule_fires_when_the_first_objective_never_lands():
    config = TrainConfig(kill_after_hours=3.0, kill_unless="get_parcel")
    assert should_stop(elapsed_hours=3.5, completed=set(), config=config) == "kill-rule"
    assert should_stop(elapsed_hours=3.5, completed={"get_parcel"}, config=config) is None


def test_the_wall_clock_budget_stops_the_run():
    config = TrainConfig(max_hours=9.0)
    assert should_stop(elapsed_hours=9.1, completed={"get_parcel"}, config=config) == "time"
    assert should_stop(elapsed_hours=8.9, completed={"get_parcel"}, config=config) is None


def test_reaching_the_forest_does_not_stop_training():
    """Training keeps going after a success; eval is what scores the policy."""
    config = TrainConfig(max_hours=9.0)
    assert should_stop(1.0, {"get_parcel", "deliver_parcel", "enter_forest"}, config) is None


def test_a_checkpoint_round_trips(tmp_path):
    from scripts.train_rl import load_checkpoint, save_checkpoint
    from poke.rl.policy import ActorCritic
    policy = ActorCritic(feature_dim=4, num_frames=2, hidden=8)
    optimizer = torch.optim.Adam(policy.parameters())
    path = tmp_path / "last.pt"
    save_checkpoint(path, policy, optimizer, updates=7, steps=123, completed={"get_parcel"})
    fresh = ActorCritic(feature_dim=4, num_frames=2, hidden=8)
    state = load_checkpoint(path, fresh, torch.optim.Adam(fresh.parameters()))
    assert state["updates"] == 7 and state["steps"] == 123
    assert state["completed"] == {"get_parcel"}
    for a, b in zip(policy.parameters(), fresh.parameters()):
        assert torch.allclose(a, b)


def test_new_cells_are_snapshotted_into_the_archive():
    handles = [FakeHandle(map_id=12), FakeHandle(map_id=42)]
    driver = trainer(handles)
    driver.collect(driver.new_buffer())
    assert len(driver.archive) >= 2, "two maps, two cells at least"
    assert any(message[0] == "snapshot" for message in handles[0].sent)


def test_every_worker_is_stepped_before_any_is_waited_on():
    """Send-then-wait per worker serialises the emulators: worker 1 only starts
    after worker 0 has finished its 60 frames. Measured at 60 steps/s against
    a 400 steps/s floor."""
    log = []
    handles = [OrderedHandle("a", log), OrderedHandle("b", log)]
    driver = trainer(handles, rollout_steps=1)
    driver.collect(driver.new_buffer())
    steps = [entry for entry in log
             if entry[0] == "recv" or (entry[0] == "send" and entry[2] == "step")]
    first_step = next(i for i, e in enumerate(steps) if e[0] == "send")
    window = steps[first_step:first_step + 4]
    assert [e[0] for e in window] == ["send", "send", "recv", "recv"], window


def test_the_archive_lives_beside_the_checkpoint():
    from scripts.train_rl import archive_path
    assert archive_path("checkpoints/rl").name == "archive.pkl"
    assert archive_path("checkpoints/rl").parent.name == "rl"


def test_a_resumed_driver_keeps_counting_from_where_it_stopped():
    """The first resume restarted the step counter at zero, so the log said
    8,192 steps for a run that had already done 747,520."""
    from scripts.train_rl import apply_resume
    driver = trainer([FakeHandle()])
    assert driver.steps == 0
    apply_resume(driver, {"steps": 747520, "completed": {"get_parcel"}, "updates": 730})
    assert driver.steps == 747520
    assert driver.completed == {"get_parcel"}
    driver.collect(driver.new_buffer())
    assert driver.steps == 747520 + 4, "the resumed run adds to the old total"


def test_the_rollout_records_the_hidden_state_it_started_from():
    """Episodes outlive rollouts, so the GRU state a rollout begins from is
    the one the previous rollout ended on -- never zero. PPO replayed from
    zero for the whole 9-hour run, which makes the first epoch's importance
    ratio something other than 1 and the trust region meaningless."""
    driver = trainer([FakeHandle()])
    driver.collect(driver.new_buffer())
    carried = driver.hidden.clone()
    assert not torch.allclose(carried, torch.zeros_like(carried))
    driver.collect(driver.new_buffer())
    assert torch.allclose(driver.rollout_hidden0, carried), \
        "the second rollout starts from where the first ended"
