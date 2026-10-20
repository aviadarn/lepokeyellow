from scripts.prune_route import find_loop, shortcut_all


def test_find_loop_returns_none_without_a_repeat():
    assert find_loop([(0,), (1,), (2,), (3,)]) is None
    assert find_loop([]) is None
    assert find_loop([(0,)]) is None


def test_find_loop_takes_the_widest_span_of_the_earliest_repeat():
    assert find_loop([(0,), (1,), (2,), (1,), (3,), (1,), (4,)]) == (1, 5)


def test_find_loop_sees_a_single_repeat():
    assert find_loop([(0,), (1,), (0,)]) == (0, 2)


def test_a_marked_signature_no_longer_matches():
    """A rejected cut tags the signature so the same loop is not retried
    forever; the tagged entry must stop comparing equal."""
    assert find_loop([(0,), (1,), (1,) + ("kept",)]) is None


def test_shortcut_is_none_when_there_is_nothing_to_remove():
    assert shortcut_all([(0,), (1,), (2,)]) is None


def test_shortcut_removes_a_simple_loop():
    """Signatures 0,1,2,1,3: standing on 1 twice means actions 1 and 2 were
    wasted, so keep action 0 then jump from the second visit to 1."""
    assert shortcut_all([(0,), (1,), (2,), (1,), (3,)]) == [0, 3]


def test_shortcut_removes_several_loops_in_one_pass():
    """Steps 0 and 2 share a signature, as do 4 and 6, so actions 0-1 and 4-5
    are both wasted. Jumping to the last visit of each leaves 2 -> 3 -> 4,
    then a jump to 6, then 6 -> 7: actions 2, 3 and 6."""
    signatures = [(0,), (1,), (0,), (1,), (2,), (3,), (2,), (4,)]
    assert shortcut_all(signatures) == [2, 3, 6]


def test_shortcut_keeps_a_loop_free_route_intact():
    assert shortcut_all([(0,), (1,), (2,), (3,), (4,)]) is None


def test_shortcut_handles_an_immediate_repeat():
    """A noop or a walk into a wall leaves the signature unchanged."""
    assert shortcut_all([(0,), (0,), (0,), (1,)]) == [2]


class FakeDriver:
    """Driver stand-in for the control probe: a scripted screen per button."""

    def __init__(self, screens, positions=None, start=(38, 6, 3)):
        self.screens = screens
        self.positions = positions or {}
        self.start = start
        self._position = start
        self._screen = b"start"

    def snapshot(self):
        return b"blob"

    def restore(self, blob):
        self._position, self._screen = self.start, b"start"

    @property
    def position(self):
        return self._position

    def screen_digest(self):
        return self._screen

    def act(self, button):
        self._screen = self.screens.get(button, b"same")
        self._position = self.positions.get(button, self.start)


def test_control_probe_rejects_an_open_text_box():
    """Every button feeds the same box, so both screens come out identical and
    nothing moves. This is the state the old probe cleared, leaving every save
    state and dataset starting mid-dialogue."""
    from scripts.bootstrap import responds_to_direction

    driver = FakeDriver(screens={"down": b"box2", "right": b"box2"})
    assert not responds_to_direction(driver)


def test_control_probe_rejects_a_cutscene_that_happens_to_move_the_sprite():
    """A scripted walk can move the player during the probe press. Identical
    screens for two different buttons give it away."""
    from scripts.bootstrap import responds_to_direction

    driver = FakeDriver(
        screens={"down": b"scene", "right": b"scene"},
        positions={"down": (38, 7, 3), "right": (38, 7, 3)},
    )
    assert not responds_to_direction(driver)


def test_control_probe_accepts_real_control():
    from scripts.bootstrap import responds_to_direction

    driver = FakeDriver(
        screens={"down": b"facing-down", "right": b"facing-right"},
        positions={"down": (38, 7, 3), "right": (38, 6, 4)},
    )
    assert responds_to_direction(driver)


def test_control_probe_requires_movement_not_just_a_changed_screen():
    """Blocked both ways: the screens differ because the player turns, but
    nothing moves.

    The probe rejects this on purpose. Differing screens alone would also be
    produced by a menu cursor, and bootstrap's job is to hand back a state the
    random search can navigate from -- a corner where no direction moves is not
    one, and spending one more A press costs nothing. The cost of this choice
    is a false negative in a fully blocked corner, which is the safe direction
    to err in."""
    from scripts.bootstrap import responds_to_direction

    driver = FakeDriver(screens={"down": b"facing-down", "right": b"facing-right"})
    assert not responds_to_direction(driver)
