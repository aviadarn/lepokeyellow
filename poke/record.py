"""Capture demonstrations while something else plays the game.

Behaviour cloning needs (frame, action) pairs, and the frame has to be the
one the decision was made *from* -- the frame after the action already shows
the answer. The planner in `scripts/play.py` routes every action through one
`step()`, so wrapping its env captures the whole run without touching a line
of the planner's logic.

This is why a gameplay video cannot be the source. A video has the frames and
not the buttons, and the buttons are not recoverable from the pixels: walking
into a wall and pressing nothing look identical, so do pressing A on empty
ground and waiting. The emulator knows what was pressed. A video only knows
what it looked like.

Everything here is training-time. The contract governs what a policy may read
while it plays; a recorder writes the dataset the policy is trained on, and
the clone it produces still sees nothing but pixels.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass

import numpy as np


def pause_recording(env):
    """Suspend recording on `env` for a block, whether or not it records.

    A planner that probes by snapshotting, pressing two buttons and restoring
    plays three actions nobody chose. Call sites should not have to know
    whether they are being recorded, so this is a no-op on a plain env.
    """
    paused = getattr(env, "paused", None)
    return paused() if callable(paused) else nullcontext()


@dataclass
class Demo:
    """One decision: what was on screen, and what was pressed."""

    frame: np.ndarray
    action: int
    label: str | None = None
    location: tuple | None = None


class RecordingEnv:
    """An env that remembers every action taken through it.

    Transparent by design: anything it does not define is forwarded, because
    the planner touches a great many methods and must not notice the wrapper.
    """

    def __init__(self, env, locate=None):
        # Bypass __setattr__ indirection worries: these are the only names
        # this class owns, everything else belongs to the wrapped env.
        self._env = env
        self._steps: list[Demo] = []
        self._label: str | None = None
        self._recording = True
        self._locate = locate

    def __getattr__(self, name):
        # Only reached for names this object does not have, so the wrapper's
        # own attributes never shadow the env's.
        return getattr(self._env, name)

    def __len__(self) -> int:
        return len(self._steps)

    @property
    def steps(self) -> list[Demo]:
        return self._steps

    def label(self, label: str | None) -> None:
        """Name what the player is pursuing from here on."""
        self._label = label

    @contextmanager
    def paused(self):
        """Run the game without recording.

        A planner that searches by snapshotting and restoring plays actions it
        did not choose. Those are not demonstrations, and a clone trained on
        them learns the search, not the route.
        """
        was = self._recording
        self._recording = False
        try:
            yield self
        finally:
            self._recording = was

    def step(self, action_index: int):
        if self._recording:
            frame = self._env.screen()
            self._steps.append(Demo(
                frame=np.ascontiguousarray(frame[:, :, 0]),
                action=int(action_index),
                label=self._label,
                location=None if self._locate is None else self._locate(self._env),
            ))
        return self._env.step(action_index)
