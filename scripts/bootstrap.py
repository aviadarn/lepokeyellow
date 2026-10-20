"""From a bare ROM to a verified save state, with nobody playing.

Upstream needs a hand-made save state: you play until you are standing in
front of the right Poke Ball and close the window. That makes every result
depend on an artifact nobody else can reproduce, and it is why routes have to
carry the digest of the state they were recorded against.

None of it is necessary. Yellow's whole intro -- Oak's speech, both name
entries, the walk-on -- is cleared by pressing A, and from the bedroom a
seeded random walk finds the way out. So:

  1. press A until the game hands over control (confirmed by a probe move
     that is rolled back, so the probe cannot displace the player),
  2. random-walk with a fixed seed until the target map is reached,
  3. truncate the walk at first arrival and REPLAY it from the start to check
     it lands there again,
  4. save the state and the route.

Same seed, same ROM, same state. A second machine with the same dump gets the
same file, which is what upstream's hand-made state can never give.

  uv run python -m scripts.bootstrap --to oaks_lab
"""

from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

import numpy as np
from pyboy import PyBoy

from poke.actions import ACTIONS
from poke.games import resolve
from poke.maps import NAMES, OAKS_LAB, PALLET_TOWN, REDS_HOUSE_1F, REDS_HOUSE_2F, name
from poke.routes import SCHEMA, Route, state_digest

def _on_map(map_id):
    return lambda driver: driver.read(driver.game.ram.cur_map) == map_id


def _has_pokemon(driver):
    return driver.read(driver.game.ram.party_count) >= 1


# A target is a predicate, not just a map id. "Standing in Oak's lab" and
# "holding a Pokemon" are different things and the lab is reachable long before
# the story will hand anything over: bootstrapping to map 40 alone produces a
# state where no policy can get Pikachu, because Oak has not stopped the player
# north of Pallet yet. Every baseline scored 0/1 there, which looks like a hard
# task and is really an impossible one.
TARGETS = {
    "bedroom": _on_map(REDS_HOUSE_2F),
    "house": _on_map(REDS_HOUSE_1F),
    "pallet": _on_map(PALLET_TOWN),
    "oaks_lab": _on_map(OAKS_LAB),
    "pikachu": _has_pokemon,
}

# Which task each target produces a demonstration for. The route is stored under
# that task's name, because routes.check refuses a route recorded for a
# different task -- a guard that caught this exact mix-up.
TARGET_TASK = {
    "bedroom": None,
    "house": "leave_room",
    "pallet": "leave_room",
    "oaks_lab": "leave_room",
    "pikachu": "yellow_pikachu",
}

# Buttons the search may use. "start" is excluded: the menu is a trap a random
# walk never escapes, and the model's action space does not contain it either.
WALK_BUTTONS = ("up", "down", "left", "right", "a", "b")


class Driver:
    """Thin PyBoy wrapper matching PokeEnv's action timing exactly."""

    def __init__(self, rom: str, game_key: str | None = None, act_frames: int = 60):
        self.game = resolve(rom, game_key)
        # DMG, for the same reason PokeEnv forces it: channel 0 must be luminance.
        self.pyboy = PyBoy(rom, window="null", cgb=False)
        self.hold = act_frames // 3
        self.settle = act_frames - self.hold

    def boot(self) -> None:
        self.pyboy.tick(300, True)

    def act(self, button: str | None) -> None:
        if button:
            self.pyboy.button_press(button)
        self.pyboy.tick(self.hold, render=False)
        if button:
            self.pyboy.button_release(button)
        self.pyboy.tick(self.settle, render=True)

    def read(self, addr: int) -> int:
        return self.pyboy.memory[addr]

    @property
    def position(self) -> tuple[int, int, int]:
        ram = self.game.ram
        return (self.read(ram.cur_map), self.read(ram.y_coord), self.read(ram.x_coord))

    def snapshot(self) -> bytes:
        buffer = io.BytesIO()
        self.pyboy.save_state(buffer)
        return buffer.getvalue()

    def restore(self, blob: bytes) -> None:
        self.pyboy.load_state(io.BytesIO(blob))
        self.pyboy.tick(1, True)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            self.pyboy.save_state(handle)

    def close(self) -> None:
        self.pyboy.stop()


def responds_to_direction(driver: Driver) -> bool:
    """True only if a direction moves the player and the reverse brings it back.

    Two traps this guards against. During Yellow's walk-on the cutscene moves
    the sprite with no input at all, so "position changed" alone reports
    control that does not exist -- hence the no-op with identical timing first.
    And a scripted walk could coincidentally match one direction, so the move
    has to be reversible, which no cutscene will honour.

    Presses real buttons; the caller sandboxes it.
    """
    before = driver.position
    driver.act(None)
    if driver.position != before:
        return False  # something else is driving the sprite

    for button, reverse in (("down", "up"), ("right", "left")):
        driver.act(button)
        if driver.position == before:
            continue
        driver.act(reverse)
        if driver.position == before:
            return True
    return False


def has_control(driver: Driver) -> bool:
    """Is the player controllable right now? Side-effect free.

    Snapshot, probe, restore. Snapshot fidelity was checked directly: position,
    screen and WRAM all come back identical, and replaying the same buttons
    from a restored state is deterministic.
    """
    blob = driver.snapshot()
    try:
        return responds_to_direction(driver)
    finally:
        driver.restore(blob)


def start_new_game(driver: Driver, max_presses: int = 400) -> list[str]:
    """Press A until the game hands over control.

    Only A is ever committed. An earlier version pressed B speculatively to
    close the trailing text box and burned 5,104 actions: B is *backspace* on
    Yellow's name-entry screen, so it bounces between the name menu and the
    letter grid forever. A is safe everywhere -- in the bedroom it opens
    whatever the player faces and the next A closes it again, so the probe
    succeeds on an even press.
    """
    actions: list[str] = []
    for _ in range(max_presses):
        driver.act("a")
        actions.append("a")
        if driver.read(driver.game.ram.cur_map) != REDS_HOUSE_2F:
            continue
        if has_control(driver):
            return actions
    raise RuntimeError(
        f"no control after {len(actions)} A presses (map "
        f"{driver.read(driver.game.ram.cur_map)}). The intro differs on this ROM."
    )


def search_to_goal(driver: Driver, reached, seed: int, max_steps: int):
    """Random-walk until `reached(driver)`. Returns (actions, steps) or None."""
    rng = np.random.default_rng(seed)
    actions: list[str] = []
    for step in range(1, max_steps + 1):
        button = WALK_BUTTONS[rng.integers(0, len(WALK_BUTTONS))]
        driver.act(button)
        actions.append(button)
        if reached(driver):
            return actions, step
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rom", default="pokeyellow.gbc")
    parser.add_argument("--to", default="oaks_lab", choices=sorted(TARGETS))
    parser.add_argument("--out")
    parser.add_argument("--route")
    parser.add_argument("--task", help="task the route is a demonstration for")
    parser.add_argument("--seeds", type=int, default=8,
                        help="search seeds to try; the shortest route wins")
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--act-frames", type=int, default=60)
    parser.add_argument("--game")
    args = parser.parse_args()

    reached = TARGETS[args.to]
    out = Path(args.out or f"states/{args.to}.state")
    route_task = args.task or TARGET_TASK[args.to] or f"bootstrap_{args.to}"
    route_path = Path(args.route or f"routes/{route_task}.json")

    driver = Driver(args.rom, args.game, args.act_frames)
    started = time.perf_counter()
    try:
        driver.boot()
        intro = start_new_game(driver)
        print(f"intro cleared in {len(intro)} A presses; now at {driver.position} "
              f"-- {name(driver.position[0])}")
        bedroom = driver.snapshot()
        # Always write the post-intro state, whatever the target. The search
        # route replays FROM it, so it has to exist on disk for the route's
        # digest to mean anything.
        bedroom_path = Path("states/bedroom.state")
        driver.save(bedroom_path)

        if reached(driver):
            best, best_seed = [], None
        else:
            best, best_seed = None, None
            for seed in range(args.seeds):
                driver.restore(bedroom)
                found = search_to_goal(driver, reached, seed, args.max_steps)
                if found is None:
                    print(f"  seed {seed}: not reached in {args.max_steps} steps")
                    continue
                actions, steps = found
                print(f"  seed {seed}: {args.to} in {steps} steps")
                if best is None or len(actions) < len(best):
                    best, best_seed = actions, seed
            if best is None:
                raise SystemExit(
                    f"no seed reached {args.to!r} within {args.max_steps} steps. "
                    "Raise --seeds or --max-steps."
                )
            print(f"shortest: seed {best_seed}, {len(best)} actions")

        # Replay from the bedroom and check it lands in the same place. A route
        # that only worked inside the search loop is a route that does not work.
        driver.restore(bedroom)
        for button in best:
            driver.act(button)
        if not reached(driver):
            raise SystemExit(
                f"replay did not reach {args.to!r} (ended at {driver.position}). "
                "The route is not reproducible; this is a bug, not a retry."
            )
        print(f"replay verified: {driver.position} -- {name(driver.position[0])}, "
              f"party={driver.read(driver.game.ram.party_count)}")

        driver.save(out)
    finally:
        driver.close()

    # The route carries the digest of the state its actions START from, not the
    # one they end at. The intro is not part of it: those presses run from a
    # bare boot, and their result is bedroom.state itself.
    if best:
        Route(
            schema=SCHEMA,
            game=driver.game.key,
            task=args.task or TARGET_TASK[args.to] or f"bootstrap_{args.to}",
            state_sha1=state_digest(bedroom_path),
            actions=best,
            reached_goal=True,
            goal_step=len(best),
        ).save(route_path)

    elapsed = time.perf_counter() - started
    print(f"\nwrote {bedroom_path}  ({state_digest(bedroom_path)[:12]})  "
          f"{len(intro)} intro actions from boot")
    if out != bedroom_path:
        print(f"wrote {out}  ({state_digest(out)[:12]})")
    if best:
        print(f"wrote {route_path}  {len(best)} actions, replayed from "
              f"{bedroom_path.name}")
    print(f"{elapsed:.1f}s. Same seed and ROM reproduce this exactly.")
    unknown = [a for a in intro + best if a not in ACTIONS]
    assert not unknown, unknown


if __name__ == "__main__":
    main()
