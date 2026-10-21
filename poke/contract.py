"""The play-time contract, as code that raises.

At training time anything goes: RAM rewards, save-state resets, oracle routes.
At play time the policy sees pixels, the frozen encoder's embeddings and which
objective it is on, and nothing else. In particular it may not read the map id
or the coordinates, and the run may not roll back.

Those are anti-goals in the design document. Anti-goals that are only written
down get broken quietly, so `PlayTimeEnv` turns them into exceptions. The
scorer -- `Run` -- keeps the unguarded env, because judging a run is allowed to
know where the player is.
"""

from __future__ import annotations

from poke.actions import ACTIONS
from poke.objectives import current


class ContractViolation(Exception):
    pass


class PlayTimeEnv:
    """Everything `PokeEnv` does, minus what the contract forbids."""

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        if name in ("snapshot", "restore"):
            raise ContractViolation(
                f"{name}() is rollback, which play time bans: the run is one "
                "continuous episode. Use it in training code only."
            )
        return getattr(self._env, name)

    def read_ram(self, addr):
        ram = self._env.game.ram
        if addr in (ram.cur_map, ram.y_coord, ram.x_coord):
            raise ContractViolation(
                f"{addr:#06x} is position. The policy navigates from pixels; "
                "map id and coordinates are for scoring, on the unguarded env."
            )
        return self._env.read_ram(addr)


class Run:
    """One continuous episode's log, controller tally and tile coverage.

    Holds the unguarded env on purpose: this is the scorer.
    """

    #: Controllers whose actions are neither learned nor handler work.
    UNOWNED = ("no_control",)

    def __init__(self, env):
        self.env = env
        self.log: list[dict] = []
        self.by_controller: dict[str, int] = {}
        self.tiles_seen: set[tuple] = set()
        self.objective_key = current(env).key
        self.new_tiles_this_objective = 0

    def _tile(self):
        ram = self.env.game.ram
        return (self.env.read_ram(ram.cur_map), self.env.read_ram(ram.y_coord),
                self.env.read_ram(ram.x_coord))

    def record(self, action_index: int, controller: str) -> None:
        """Called once per emulator action, after it has been taken."""
        objective = current(self.env)
        if objective.key != self.objective_key:
            # A new objective re-opens the whole map: the walk back to Oak is
            # new ground again, which is what makes the backtrack pay.
            self.objective_key = objective.key
            self.tiles_seen = set()
            self.new_tiles_this_objective = 0
        tile = self._tile()
        if tile not in self.tiles_seen:
            self.tiles_seen.add(tile)
            self.new_tiles_this_objective += 1
        self.by_controller[controller] = self.by_controller.get(controller, 0) + 1
        self.log.append({
            "n": len(self.log) + 1,
            "objective": objective.key,
            "controller": controller,
            "action": ACTIONS[action_index],
            "map": tile[0], "y": tile[1], "x": tile[2],
        })

    @property
    def learned_share(self) -> float:
        """Policy actions over policy plus handler actions.

        Turns nobody had -- the input the fade-in ate -- are excluded from both
        sides rather than counted against the policy. Note the measured limit
        recorded in poke.control.MEASURED: the text-box detector is right on
        about 95% of turns, and a walking cutscene with no box on screen is
        handed to the policy and counted here, so this number is an upper
        bound rather than an exact share.
        """
        owned = {k: v for k, v in self.by_controller.items() if k not in self.UNOWNED}
        total = sum(owned.values())
        return 0.0 if total == 0 else owned.get("policy", 0) / total
