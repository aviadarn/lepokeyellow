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
