"""Train the Part 2 policy. One arm: frozen world-model features, world-model bonus.

Four worker processes each hold a PyBoy and a TrainEnv; the driver holds the
encoder, the policy, the archive and PPO. Frames cross the pipe (23 KB each,
about 55 MB/s at the measured rate) because the encoder lives on MPS in the
driver and four processes sharing one GPU queue is worse than the copy.

What the driver owns and the workers do not: the archive. A worker reports the
cell it is standing in; when that cell is new the driver asks for a snapshot.
Blobs are 178 KB, so asking only on a new cell is the difference between a few
megabytes an hour and a few gigabytes.

Episodes outlive rollouts. A training episode is 512 actions and a rollout is
128, so the driver seeds from the archive once and then carries each worker's
episode across rollouts, resetting a column only when that episode actually
ends. Reseeding every rollout would mean no episode ever finished and the
policy never saw a consequence more than 128 actions away.

Stopping is wall-clock, not steps. The step rate is an estimate; the overnight
window is not. The kill rule from the design is separate and earlier: if no
episode has completed `get_parcel` by --kill-after-hours, the reward or the
archive is wrong and more hours will not fix it.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from multiprocessing import Pipe, Process
from pathlib import Path

import numpy as np
import torch

from poke.goals import default_device
from poke.objectives import N_OBJECTIVES
from poke.rl.archive import CELL, Archive, Cell
from poke.rl.features import EmbeddingStack, FrozenEncoder
from poke.rl.intrinsic import CHOSEN, WorldModelBonus
from poke.rl.policy import ActorCritic, observation
from poke.rl.ppo import PPOConfig, RolloutBuffer, ppo_update
from poke.rl.trainenv import TrainEnv


@dataclass
class TrainConfig:
    rom: str = "pokeyellow.gbc"
    state: str = "states/oracle/route1.state"
    checkpoint: str = "checkpoints/part2/best.pt"
    out_dir: str = "checkpoints/rl"
    workers: int = 4
    rollout_steps: int = 128
    episode_len: int = 512
    bonus_coef: float = 0.05
    max_hours: float = 9.0
    kill_after_hours: float | None = 3.0
    kill_unless: str = "get_parcel"
    checkpoint_every: int = 10
    #: Updates between archive writes. It is 50 MB, so less often than weights.
    archive_every: int = 100
    seed: int = 0
    ppo: PPOConfig = field(default_factory=PPOConfig)


def should_stop(elapsed_hours: float, completed: set, config: TrainConfig) -> str | None:
    """Why training should end, or None. Pure, so the rule is testable."""
    if elapsed_hours >= config.max_hours:
        return "time"
    if (config.kill_after_hours is not None
            and elapsed_hours >= config.kill_after_hours
            and config.kill_unless not in completed):
        return "kill-rule"
    return None


# -- the worker process ----------------------------------------------------

def worker_loop(pipe, rom, state, episode_len):
    """One emulator. It answers; the driver decides."""
    from poke.control import has_control
    env = TrainEnv(rom=rom, state_path=state, episode_len=episode_len)
    step = env.reset()
    while True:
        message, payload = pipe.recv()
        if message == "close":
            env.env.close()
            return
        if message == "snapshot":
            pipe.send(env.snapshot())
            continue
        if message == "reset":
            step = env.reset(payload)
        elif message == "step":
            step = env.step(int(payload))
        info = dict(step.info)
        info["can_snapshot"] = has_control(env.env)
        pipe.send({"frame": step.frame, "objective": step.objective,
                   "reward": step.reward, "done": step.done, "info": info})


class WorkerHandle:
    """One worker process and its pipe, restartable."""

    def __init__(self, rom, state, episode_len):
        self.spec = (rom, state, episode_len)
        self.process = None
        self.pipe = None
        self.restart()

    def restart(self):
        if self.process is not None and self.process.is_alive():
            self.process.terminate()
        parent, child = Pipe()
        self.pipe = parent
        self.process = Process(target=worker_loop, args=(child, *self.spec), daemon=True)
        self.process.start()

    def send(self, message):
        self.pipe.send(message)

    def recv(self):
        return self.pipe.recv()

    def close(self):
        try:
            self.send(("close", None))
        except (BrokenPipeError, OSError):
            pass


# -- the driver ------------------------------------------------------------

BLANK_FRAME = np.zeros((144, 160), dtype=np.uint8)


class Trainer:
    def __init__(self, handles, policy, encoder, archive, bonus, config, device="cpu"):
        self.handles = list(handles)
        self.policy = policy
        self.encoder = encoder
        self.archive = archive
        self.bonus = bonus
        self.config = config
        self.device = device
        self.stacks = [EmbeddingStack(encoder.num_frames, encoder.dim) for _ in handles]
        self.prev_action = [0] * len(handles)
        self.hidden = policy.initial_hidden(len(handles), device)
        self.rollout_hidden0 = self.hidden.clone()
        self.completed: set[str] = set()
        self.steps = 0
        self.episodes = 0
        self.last_obs: torch.Tensor | None = None
        self.last_embeddings: np.ndarray | None = None

    def new_buffer(self) -> RolloutBuffer:
        return RolloutBuffer(self.config.rollout_steps, len(self.handles),
                             self.policy.obs_dim, device=self.device)

    # -- observation -------------------------------------------------------

    def _blank(self) -> dict:
        return {"frame": BLANK_FRAME, "objective": [0.0] * N_OBJECTIVES,
                "reward": 0.0, "done": False,
                "info": {"map": None, "y": 0, "x": 0, "completed": [],
                         "controllers": {}, "taken": 0, "can_snapshot": False}}

    def _row(self, index: int, embedding, objective, first: bool) -> np.ndarray:
        stack = self.stacks[index]
        if first:
            stack.reset(embedding)
        else:
            stack.push(embedding)
        return observation(stack.vector(), objective, self.prev_action[index],
                           self.policy.n_actions)

    def _tensor(self, rows) -> torch.Tensor:
        return torch.from_numpy(np.stack(rows)).float().to(self.device)

    def _send(self, index: int, message, dropped: set) -> None:
        """Hand one worker its next job. Never waits."""
        if index in dropped:
            return
        try:
            self.handles[index].send(message)
        except (EOFError, OSError, BrokenPipeError):
            self.handles[index].restart()
            dropped.add(index)

    def _wait(self, index: int, dropped: set):
        """Collect one worker's answer, surviving its death."""
        if index in dropped:
            return self._blank()
        try:
            return self.handles[index].recv()
        except (EOFError, OSError, BrokenPipeError):
            self.handles[index].restart()
            dropped.add(index)
            return self._blank()

    def _ask(self, index: int, message, dropped: set):
        """Send and wait, for the one-off calls: reset, snapshot."""
        self._send(index, message, dropped)
        return self._wait(index, dropped)

    # -- the archive -------------------------------------------------------

    def _note_cell(self, index: int, reply: dict, dropped: set) -> None:
        if index in dropped or not reply["info"].get("can_snapshot"):
            return
        info = reply["info"]
        key = (int(np.argmax(reply["objective"])), info["map"],
               (info["y"] or 0) // CELL, (info["x"] or 0) // CELL)
        if key in self.archive.cells:
            self.archive.note_visit(key)
            return
        blob = self._ask(index, ("snapshot", None), dropped)
        if isinstance(blob, bytes):
            self.archive.cells[key] = Cell(key=key, blob=blob,
                                           found_at=len(self.archive) + 1)

    # -- rollout -----------------------------------------------------------

    def _start_episode(self, index: int, dropped: set) -> np.ndarray | None:
        """Reset one worker from the archive and clear its memory."""
        reply = self._ask(index, ("reset", self.archive.sample()), dropped)
        embedding = self.encoder.embed(reply["frame"][None, ...])[0]
        self.prev_action[index] = 0
        with torch.no_grad():
            self.hidden[index] = 0.0
        return self._row(index, embedding, reply["objective"], first=True), embedding

    def collect(self, buffer: RolloutBuffer) -> dict:
        """One rollout. Returns which env columns are usable for the update."""
        dropped: set[int] = set()

        if self.last_obs is None:
            rows, embeddings = [], []
            for index in range(len(self.handles)):
                row, embedding = self._start_episode(index, dropped)
                rows.append(row)
                embeddings.append(embedding)
            obs = self._tensor(rows)
            previous = np.stack(embeddings)
        else:
            obs, previous = self.last_obs, self.last_embeddings

        # The hidden state this rollout is generated FROM. PPO's importance
        # ratio is only 1 on the first epoch if evaluate() replays from this
        # exact state; episodes outlive rollouts, so it is not zero.
        self.rollout_hidden0 = self.hidden.detach().clone()

        for t in range(buffer.steps):
            with torch.no_grad():
                out = self.policy.act(obs, self.hidden)
            self.hidden = out["hidden"]
            actions = out["action"].tolist()

            # Send to every worker before waiting on any of them. Sending and
            # waiting one at a time serialises the emulators: worker 1 would
            # only start its 60 frames once worker 0 had finished. Measured at
            # 60 driver steps/s that way, against a 400 floor.
            for index in range(len(self.handles)):
                self._send(index, ("step", actions[index]), dropped)
            replies = [self._wait(index, dropped) for index in range(len(self.handles))]
            embeddings = self.encoder.embed(np.stack([r["frame"] for r in replies]))

            rewards = [float(reply["reward"]) for reply in replies]
            if self.bonus is not None:
                # Policy transitions only: a handler turn is a text box, and a
                # text box is a noisy TV. One forward for the whole step.
                extra = self.bonus.bonus_many(previous, actions, embeddings)
                rewards = [r + (0.0 if i in dropped else extra[i])
                           for i, r in enumerate(rewards)]
            dones = []
            for index, reply in enumerate(replies):
                dones.append(float(reply["done"]))
                self.completed.update(reply["info"]["completed"])
                self.prev_action[index] = actions[index]

            buffer.add(t, obs=obs, action=out["action"], logprob=out["logprob"],
                       value=out["value"],
                       reward=torch.tensor(rewards, device=self.device),
                       done=torch.tensor(dones, device=self.device))

            rows, next_embeddings = [], []
            for index, reply in enumerate(replies):
                self._note_cell(index, reply, dropped)
                if reply["done"]:
                    # The episode ended. Start the next one from the archive,
                    # with the frame window and the GRU state cleared: the
                    # first frame of a new episode has no past.
                    self.episodes += 1
                    row, embedding = self._start_episode(index, dropped)
                else:
                    embedding = embeddings[index]
                    row = self._row(index, embedding, reply["objective"], first=False)
                rows.append(row)
                next_embeddings.append(embedding)

            obs, previous = self._tensor(rows), np.stack(next_embeddings)
            self.steps += len(self.handles)

        self.last_obs, self.last_embeddings = obs, previous
        usable = [i for i in range(len(self.handles)) if i not in dropped]
        return {"dropped_workers": sorted(dropped), "usable_envs": usable}


def archive_path(out_dir) -> Path:
    """Where the run's curriculum lives, beside its weights."""
    return Path(out_dir) / "archive.pkl"


def apply_resume(driver, state) -> None:
    """Carry a checkpoint's progress into a fresh driver.

    The step count is part of that progress. Without it a resumed run's log
    restarts at zero, which reads as a run that has done nothing when it has
    done three quarters of a million steps.
    """
    driver.completed = set(state["completed"])
    driver.steps = int(state["steps"])


def save_checkpoint(path, policy, optimizer, updates, steps, completed):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"policy": policy.state_dict(), "optimizer": optimizer.state_dict(),
                "updates": updates, "steps": steps, "completed": sorted(completed)}, path)


def load_checkpoint(path, policy, optimizer) -> dict:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    policy.load_state_dict(blob["policy"])
    optimizer.load_state_dict(blob["optimizer"])
    return {"updates": blob["updates"], "steps": blob["steps"],
            "completed": set(blob["completed"])}


def run_report(reason, updates, steps, episodes, hours, completed, archive,
               history, resumed_from=None, bonus=None, encoder=None) -> dict:
    """What the run is allowed to claim about itself.

    A resumed run inherits the update counter from the checkpoint and starts
    its step counter at zero, so `updates` spans every run that produced
    these weights while `steps` and `hours` describe this one. Reporting the
    two side by side made the headline "8,941 PPO updates in nine hours",
    when 8,211 of them happened in those hours.

    The history is every row. It used to be the last 200 of 8,941 -- twelve
    minutes of a nine-hour run -- and trends were then quoted from a window
    too short to contain them.
    """
    before = (resumed_from or {}).get("updates", 0)
    return {
        "stopped": reason,
        "updates": updates,
        "updates_this_run": updates - before,
        "steps": steps,
        "steps_this_run": steps,
        "episodes": episodes,
        "hours": hours,
        "resumed": resumed_from is not None,
        "resumed_from": resumed_from,
        "completed": sorted(completed),
        "bonus": bonus,
        "encoder": encoder,
        "archive": archive,
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = TrainConfig()
    parser.add_argument("--rom", default=defaults.rom)
    parser.add_argument("--state", default=defaults.state)
    parser.add_argument("--checkpoint", default=defaults.checkpoint,
                        help="the frozen encoder")
    parser.add_argument("--out-dir", default=defaults.out_dir)
    parser.add_argument("--workers", type=int, default=defaults.workers)
    parser.add_argument("--rollout-steps", type=int, default=defaults.rollout_steps)
    parser.add_argument("--episode-len", type=int, default=defaults.episode_len)
    parser.add_argument("--bonus-coef", type=float, default=defaults.bonus_coef)
    parser.add_argument("--max-hours", type=float, default=defaults.max_hours)
    parser.add_argument("--kill-after-hours", type=float, default=defaults.kill_after_hours)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--half", action="store_true",
                        help="fp16 encoder: the driver's cost per tick is one "
                             "forward of it")
    parser.add_argument("--log", default="results/part2_rl_train.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    config = TrainConfig(rom=args.rom, state=args.state, checkpoint=args.checkpoint,
                         out_dir=args.out_dir, workers=args.workers,
                         rollout_steps=args.rollout_steps, episode_len=args.episode_len,
                         bonus_coef=args.bonus_coef, max_hours=args.max_hours,
                         kill_after_hours=args.kill_after_hours, seed=args.seed)

    encoder = FrozenEncoder(config.checkpoint, device=args.device, half=args.half)
    policy = ActorCritic(feature_dim=encoder.dim, num_frames=encoder.num_frames).to(args.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.ppo.lr)
    bonus = WorldModelBonus(encoder.model, device=args.device, coef=config.bonus_coef)
    # The archive is the curriculum: restore it with the weights, or a resumed
    # run keeps what the policy learned and forgets where it learned it.
    archive = Archive.load(archive_path(config.out_dir)) if args.resume else Archive()
    if len(archive):
        print(f"restored archive: {archive.stats()}", flush=True)
    handles = [WorkerHandle(config.rom, config.state, config.episode_len)
               for _ in range(config.workers)]

    state = {"updates": 0, "steps": 0, "completed": set()}
    resumed_from = None
    if args.resume:
        state = load_checkpoint(args.resume, policy, optimizer)
        # What this run inherited, so the report can separate it from what
        # this run did. The counters mean different things: updates carries
        # over, steps does not.
        resumed_from = {"updates": int(state.get("updates", 0)),
                        "steps": int(state.get("steps", 0)),
                        "completed": sorted(state.get("completed", set())),
                        "checkpoint": args.resume}

    driver = Trainer(handles, policy, encoder, archive, bonus, config, device=args.device)
    apply_resume(driver, state)
    started = time.time()
    history = []
    reason = None

    try:
        while reason is None:
            buffer = driver.new_buffer()
            report = driver.collect(buffer)
            with torch.no_grad():
                _, last_value, _ = policy(driver.last_obs, driver.hidden)
            buffer.compute_returns(last_value, torch.zeros_like(last_value), config.ppo)
            stats = ppo_update(policy, optimizer, buffer, config.ppo,
                               driver.rollout_hidden0, usable=report["usable_envs"])
            state["updates"] += 1
            elapsed = (time.time() - started) / 3600.0
            row = {"update": state["updates"], "steps": driver.steps,
                   "episodes": driver.episodes, "hours": round(elapsed, 3),
                   "steps_per_s": round(driver.steps / max(time.time() - started, 1e-9), 1),
                   "archive": archive.stats(), "completed": sorted(driver.completed),
                   "dropped_workers": report["dropped_workers"],
                   **{k: round(v, 5) for k, v in stats.items()}}
            history.append(row)
            print(json.dumps(row), flush=True)
            if state["updates"] % config.checkpoint_every == 0:
                save_checkpoint(Path(config.out_dir) / "last.pt", policy, optimizer,
                                state["updates"], driver.steps, driver.completed)
            if state["updates"] % config.archive_every == 0:
                # Bigger than the checkpoint and slower to write, so less
                # often: 278 cells is about 50 MB.
                archive.save(archive_path(config.out_dir))
            reason = should_stop(elapsed, driver.completed, config)
    finally:
        save_checkpoint(Path(config.out_dir) / "last.pt", policy, optimizer,
                        state["updates"], driver.steps, driver.completed)
        archive.save(archive_path(config.out_dir))
        for handle in handles:
            handle.close()

    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log).write_text(json.dumps(run_report(
        reason=reason,
        updates=state["updates"],
        steps=driver.steps,
        episodes=driver.episodes,
        hours=round((time.time() - started) / 3600.0, 3),
        completed=driver.completed,
        archive=archive.stats(),
        history=history,
        resumed_from=resumed_from,
        bonus=CHOSEN["signal"],
        encoder=config.checkpoint,
    ), indent=1))
    print("stopped:", reason, "completed:", sorted(driver.completed))


if __name__ == "__main__":
    main()
