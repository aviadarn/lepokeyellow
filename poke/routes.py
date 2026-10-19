"""Demonstration routes: a human-recorded action sequence that reaches a goal.

Upstream hard-codes its routes as Python lists (`TAKE = ["a"] * 12`), which
works because its task starts the player already facing the ball. Yellow's
Pikachu scene needs navigation across the lab, and the exact tile path depends
on where the save state was made, so the route cannot be written blind.

`scripts/record_route.py` records one by hand, once, and stores it here with
the digest of the state it was recorded against. `collect.py` refuses a route
whose state digest does not match, because replaying a route from a different
save state produces confidently mislabelled episodes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path

from poke.actions import ACTIONS

SCHEMA = 1


def state_digest(state_path: str | Path) -> str:
    return hashlib.sha1(Path(state_path).read_bytes()).hexdigest()


@dataclass
class Route:
    schema: int
    game: str
    task: str
    state_sha1: str
    actions: list[str]
    reached_goal: bool
    goal_step: int

    @property
    def indices(self) -> list[int]:
        return [ACTIONS.index(name) for name in self.actions]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "Route":
        data = json.loads(Path(path).read_text())
        if data.get("schema") != SCHEMA:
            raise ValueError(
                f"{path}: route schema {data.get('schema')!r}, expected {SCHEMA}"
            )
        unknown = [name for name in data["actions"] if name not in ACTIONS]
        if unknown:
            raise ValueError(f"{path}: unknown action names {sorted(set(unknown))}")
        return cls(**data)

    def check(self, game: str, task: str, state_path: str | Path) -> None:
        """Fail loudly on a route that does not belong to this run."""
        if self.game != game:
            raise ValueError(f"route was recorded for {self.game!r}, not {game!r}")
        if self.task != task:
            raise ValueError(f"route was recorded for task {self.task!r}, not {task!r}")
        actual = state_digest(state_path)
        if self.state_sha1 != actual:
            raise ValueError(
                f"route was recorded against a different save state "
                f"({self.state_sha1[:12]} != {actual[:12]}). Re-record it with "
                "scripts/record_route.py, or the replayed episodes will be "
                "labelled wrong."
            )
        if not self.reached_goal:
            raise ValueError(
                "route is marked as never reaching the goal; collecting from it "
                "would give a dataset with no successful episodes"
            )
