import numpy as np

from poke.synth import BIT, BUTTONS, HumanLike, Pad, clean_index


def test_button_order_starts_with_the_repos_six():
    from poke.actions import BUTTON_ORDER

    assert list(BUTTONS[:6]) == BUTTON_ORDER
    assert BUTTONS[6:] == ("start", "select")


class FakePyBoy:
    def __init__(self):
        self.events = []

    def button_press(self, name):
        self.events.append(("press", name))

    def button_release(self, name):
        self.events.append(("release", name))


def test_pad_sends_only_the_changes():
    pyboy = FakePyBoy()
    pad = Pad(pyboy)
    pad.set(BIT["up"] | BIT["a"])
    pad.set(BIT["a"])
    pad.set(BIT["a"])
    assert pyboy.events == [("press", "up"), ("press", "a"), ("release", "up")]


def test_human_like_holds_then_gaps():
    policy = HumanLike(np.random.default_rng(0))
    fields = {"map_ids": 1, "ys": 5, "xs": 5, "in_battles": False, "dialog_opens": False}
    masks = [policy(lambda: fields) for _ in range(2000)]
    runs, current, length = [], masks[0], 0
    for mask in masks:
        if mask == current:
            length += 1
        else:
            runs.append((current, length))
            current, length = mask, 1
    held = [n for m, n in runs if m]
    assert np.median(held) > 3            # holds, not one-frame flicker
    assert all(0 <= m < 256 for m in masks)


def test_clean_index_marks_colours_outside_the_set():
    colours = np.array([[0, 0, 0], [248, 248, 248]], np.uint8)
    rgb = np.array([[[0, 0, 0], [248, 248, 248], [8, 16, 24]]], np.uint8)
    assert clean_index(rgb, colours).tolist() == [[0, 1, 255]]


def _a_presses(masks):
    a = np.array([(m >> BUTTONS.index("a")) & 1 for m in masks])
    return int(((a[1:] == 1) & (a[:-1] == 0)).sum())


def test_bumping_into_something_leads_to_a_presses():
    # The player never moves: every walk is a bump.
    fields = {"map_ids": 1, "ys": 5, "xs": 5, "in_battles": False, "dialog_opens": False}
    plain = HumanLike(np.random.default_rng(0))
    bumpy = HumanLike(np.random.default_rng(0), bump_a=0.8)
    plain_masks = [plain(lambda: fields) for _ in range(6000)]
    bumpy_masks = [bumpy(lambda: fields) for _ in range(6000)]
    assert _a_presses(bumpy_masks) > 3 * max(1, _a_presses(plain_masks))


def test_bump_a_off_keeps_the_old_stream():
    fields = {"map_ids": 1, "ys": 5, "xs": 5, "in_battles": False, "dialog_opens": False}
    old = HumanLike(np.random.default_rng(3))
    new = HumanLike(np.random.default_rng(3), bump_a=0.0)
    assert [old(lambda: fields) for _ in range(3000)] == [new(lambda: fields) for _ in range(3000)]


def test_menu_tapper_can_let_the_intro_play():
    from poke.synth import MenuTapper

    fields = {}
    for seed in range(5):
        tapper = MenuTapper(np.random.default_rng(seed), idle=(1500, 3600))
        masks = [tapper(lambda: fields) for _ in range(4000)]
        first = next(t for t, m in enumerate(masks) if m)
        assert 1500 <= first < 3600


def _runs(masks, bit):
    held = np.array([(m >> bit) & 1 for m in masks])
    edges = np.flatnonzero(np.diff(np.r_[0, held, 0]))
    return (edges[1::2] - edges[::2]).tolist()


def test_text_hold_holds_a_or_b_in_text_boxes():
    fields = {"map_ids": 1, "ys": 5, "xs": 5, "in_battles": False, "dialog_opens": True}
    policy = HumanLike(np.random.default_rng(0), text_hold=1.0)
    masks = [policy(lambda: fields) for _ in range(3000)]
    long_runs = [r for bit in (4, 5) for r in _runs(masks, bit) if r >= 10]
    assert len(long_runs) >= 5


def test_text_hold_off_keeps_the_old_stream():
    fields = {"map_ids": 1, "ys": 5, "xs": 5, "in_battles": False, "dialog_opens": True}
    old = HumanLike(np.random.default_rng(4))
    new = HumanLike(np.random.default_rng(4), text_hold=0.0)
    assert [old(lambda: fields) for _ in range(3000)] == [new(lambda: fields) for _ in range(3000)]


class _FakePyBoy:
    def __init__(self):
        from collections import defaultdict
        self.ticks, self.memory, self.writes = 0, defaultdict(int), 0
        self.screen = type("S", (), {"ndarray": np.zeros((144, 160, 4), np.uint8)})()

    def tick(self, n, render):
        self.ticks += n
        if self.ticks == 5:
            # A battle ends: the game reloads the map's wild data, rate and all,
            # without the map id changing.
            self.memory[0xD886] = 25

    def save_state(self, buffer):
        buffer.write(b"state")

    def button_press(self, name):
        pass

    def button_release(self, name):
        pass


class _FakeEnv:
    """Map 1 for frames 0-2, map 2 for 3-6, map 3 from 7 on."""

    def __init__(self):
        from poke.games import YELLOW
        self.game, self.pyboy = YELLOW, _FakePyBoy()

    def read_ram(self, addr):
        if addr == self.game.ram.cur_map:
            t = self.pyboy.ticks
            return 1 if t <= 3 else 2 if t <= 7 else 3
        return 0

    def read_ram_range(self, start, stop):
        return bytes(stop - start)


def test_boost_encounters_writes_the_rate():
    from poke.synth import boost_encounters

    env = _FakeEnv()
    boost_encounters(env, {"wGrassRate": (0, 0xD886)})
    assert env.pyboy.memory == {0xD886: 255}


def test_encounters_are_boosted_again_after_the_game_resets_them():
    from poke.synth import boost_encounters, play

    env = _FakeEnv()
    play(env, 10, np.random.default_rng(0), sample_every=20, policy=lambda observe: 0,
         on_frame=lambda e: boost_encounters(e, {"wGrassRate": (0, 0xD886)}))
    assert env.pyboy.memory[0xD886] == 255


def test_encounter_hook_replays_how_each_batch_was_collected():
    from poke.synth import encounter_hook

    symbols = {"wGrassRate": (0, 0xD886)}
    for mode, expected in (("every_frame", 255), ("map_change", 25)):
        env = _FakeEnv()
        hook = encounter_hook(symbols, mode)
        from poke.synth import play
        play(env, 6, np.random.default_rng(0), sample_every=20, policy=lambda observe: 0, on_frame=hook)
        # The fake game resets the rate at tick 5 without changing map: only
        # the every-frame boost writes it back (map_change is how v4's battle
        # batch was collected, and how it must be replayed).
        assert env.pyboy.memory[0xD886] == expected, mode
