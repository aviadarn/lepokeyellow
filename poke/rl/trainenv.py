"""One emulator, the handlers and the reward, as a step/reset loop.

TRAINING ONLY: it reads position for the reward and restores save states from
the archive. The same handlers run here as at play time -- `poke.handlers` is
the single declared harness -- so a policy trained here meets no surprises in
`scripts/play2.py`.

One policy action expands into the handler turns that follow it, as the design
specifies, and every one of them counts against the episode budget. A run that
spends its budget in menus has not navigated anywhere, and a policy that is
never charged for walking into a dialogue will keep doing it.

`max_handler_steps` bounds the expansion. The handlers can legitimately own
dozens of turns in a row (Oak's lecture), but an unbounded loop here means one
policy step can swallow a whole episode, and `poke.handlers.BOX_PATIENCE`
already yields a turn back every 32.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from poke.handlers import HandlerState, choose
from poke.objectives import one_hot
from poke.rl.reward import RewardState, step_reward

#: Actions per training episode, handler turns included.
EPISODE_LEN = 512
#: How many handler turns one policy action may expand into.
MAX_HANDLER_STEPS = 64
#: The objective that ends an episode by succeeding.
TERMINAL_OBJECTIVE = "enter_forest"


@dataclass
class Step:
    frame: np.ndarray
    objective: list[float]
    reward: float
    done: bool
    info: dict = field(default_factory=dict)


class TrainEnv:
    def __init__(self, rom: str | None = None, state_path: str | None = None,
                 episode_len: int = EPISODE_LEN,
                 max_handler_steps: int = MAX_HANDLER_STEPS, env=None):
        if env is None:
            from poke.env import PokeEnv      # imported late: tests run without a ROM
            env = PokeEnv(rom, state_path)
        self.env = env
        self.episode_len = episode_len
        self.max_handler_steps = max_handler_steps
        self.handlers = HandlerState()
        self.rewards = RewardState(objective_key="", credited=set())
        self.taken = 0
        self.completed: list[str] = []
        self.controllers: dict[str, int] = {}

    # -- the emulator ------------------------------------------------------

    def _map(self):
        return self.env.read_ram(self.env.game.ram.cur_map)

    def _take(self, action_index: int, controller: str) -> float:
        before = self._map()
        self.env.step(action_index)
        self.taken += 1
        reward, completed = step_reward(self.env, self.rewards)
        self.handlers.note(action_index, self._map() != before)
        self.completed.extend(completed)
        self.controllers[controller] = self.controllers.get(controller, 0) + 1
        return reward

    def _drain_handlers(self) -> tuple[float, int]:
        """Take every turn the handlers claim, up to the cap."""
        reward, steps = 0.0, 0
        while steps < self.max_handler_steps and not self._over():
            claimed = choose(self.env, self.handlers)
            if claimed is None:
                break
            action, controller = claimed
            reward += self._take(action, controller)
            steps += 1
        return reward, steps

    def _over(self) -> bool:
        return self.taken >= self.episode_len or TERMINAL_OBJECTIVE in self.completed

    def _observe(self, reward: float, handler_steps: int) -> Step:
        ram = self.env.game.ram
        return Step(
            frame=self.env.screen()[:, :, 0],
            objective=one_hot(self.env),
            reward=reward,
            done=self._over(),
            info={
                "map": self._map(),
                "y": self.env.read_ram(ram.y_coord),
                "x": self.env.read_ram(ram.x_coord),
                "completed": list(self.completed),
                "controllers": dict(self.controllers),
                "handler_steps": handler_steps,
                "taken": self.taken,
            },
        )

    # -- the loop ----------------------------------------------------------

    def reset(self, blob: bytes | None = None) -> Step:
        if blob is None:
            self.env.reset()
        else:
            self.env.restore(blob)
        self.handlers = HandlerState()
        self.rewards = RewardState.start(self.env)
        self.taken = 0
        self.completed = []
        self.controllers = {}
        # Hand the policy a frame it can act on. An archive state captured
        # mid-dialogue otherwise gives it an observation whose only legal
        # answer is the handler's.
        _, handler_steps = self._drain_handlers()
        return self._observe(reward=0.0, handler_steps=handler_steps)

    def step(self, action_index: int) -> Step:
        reward = self._take(action_index, "policy")
        drained, handler_steps = self._drain_handlers()
        return self._observe(reward=reward + drained, handler_steps=handler_steps)

    def snapshot(self) -> bytes:
        return self.env.snapshot()
