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


class ContractAttributeError(ContractViolation, AttributeError):
    """Denial of an attribute the play-time surface does not carry.

    Also an AttributeError so that `hasattr` answers False instead of
    exploding: copy, a gym wrapper and a logger all duck-type their way
    around an object, and only AttributeError is swallowed by that protocol.
    `read_ram` on a position address stays a plain ContractViolation -- that
    is a real attempt to cheat, not a feature probe.
    """


class PlayTimeEnv:
    """What a policy is allowed to see, and nothing else.

    Deny by default. The first version forwarded everything and named two
    forbidden attributes, which is a fence with the gate left open: `_env`
    reached the real object, `pyboy` reached raw memory, `read_ram_range` was
    a second unguarded read path, and `screen_of_state` rolls back internally.
    So the allowlist is the interface, and anything not on it raises.

    The wrapped env is kept off the instance dictionary, in a closure, so that
    there is no attribute to reach it through.
    """

    #: Everything a policy legitimately needs.
    ALLOWED = frozenset({"screen", "read_ram", "game", "act_frames"})

    #: Position, and the bytes beside it that stand in for position: the
    #: map-header pointer right after cur_map is a perfect map-id proxy, and
    #: the block coordinates sit two past y and x. Blocking three exact
    #: addresses is a value check an adversary walks around.
    POSITION_SPAN = 3

    def __init__(self, env):
        banned = self._banned_addresses(env)

        def read_ram(addr):
            if addr in banned:
                raise ContractViolation(
                    f"{addr:#06x} is position, or sits inside the bytes that "
                    "stand in for it. The policy navigates from pixels; map "
                    "id and coordinates are for scoring, on the real env."
                )
            return env.read_ram(addr)

        object.__setattr__(self, "_allowed", {
            "screen": env.screen,
            "read_ram": read_ram,
            "game": env.game,
            "act_frames": getattr(env, "act_frames", None),
        })

    @staticmethod
    def _banned_addresses(env) -> frozenset:
        ram = env.game.ram
        span = PlayTimeEnv.POSITION_SPAN
        return frozenset(
            base + offset
            for base in (ram.cur_map, ram.y_coord, ram.x_coord)
            for offset in range(span)
        )

    def __getattr__(self, name):
        allowed = object.__getattribute__(self, "_allowed")
        if name in allowed:
            return allowed[name]
        # AttributeError, not ContractViolation, for the dunders Python itself
        # probes -- copy, pickle and repr ask for those and must get a plain
        # "no such attribute" back.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        raise ContractAttributeError(
            f"{name!r} is not on the play-time surface "
            f"({', '.join(sorted(allowed))}). Nothing off it is reachable: "
            "not rollback, not raw memory, not the second read path."
        )

    def __setattr__(self, name, value):
        raise ContractViolation("the play-time env is read-only")


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
