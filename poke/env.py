import numpy as np
from pyboy import PyBoy

from poke.actions import to_button
from poke.games import Game, resolve


class PokeEnv:
    """PyBoy wrapper. One `step` holds a button, then lets the game settle.

    Carries the resolved `Game` so RAM reads use this ROM's addresses.
    """

    def __init__(self, rom, init_state, headless=True, act_frames=60, game=None):
        self.game: Game = resolve(rom, game)
        self.rom = rom
        self.pyboy = PyBoy(rom, window="null" if headless else "SDL2")

        self.init_state = init_state
        self.headless = headless
        self.act_frames = act_frames
        self.hold = act_frames // 3
        self.settle = act_frames - self.hold

    def reset(self) -> np.ndarray:
        with open(self.init_state, "rb") as f:
            self.pyboy.load_state(f)
        self.pyboy.tick(1, True)
        return self.screen()

    def step(self, action_idx: int) -> np.ndarray:
        button = to_button(action_idx)
        if button:
            self.pyboy.button_press(button)
        self.pyboy.tick(self.hold, render=False)
        if button:
            self.pyboy.button_release(button)
        self.pyboy.tick(self.settle, render=True)
        return self.screen()

    def read_ram(self, addr: int) -> int | None:
        try:
            return self.pyboy.memory[addr]
        except Exception:
            return None

    def read_ram_range(self, start: int, stop: int) -> bytes:
        """Inclusive-exclusive WRAM slice, for scripts/calibrate_ram.py."""
        return bytes(self.pyboy.memory[start:stop])

    def screen(self) -> np.ndarray:
        return np.asarray(self.pyboy.screen.ndarray[:, :, :3]).copy()

    def close(self) -> None:
        self.pyboy.stop()
