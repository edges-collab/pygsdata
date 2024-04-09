"""Test the history module."""
from datetime import datetime

import pytest
from pygsdata import History, Stamp


def test_history():
    history = History(
        (
            Stamp(message="hello"),
            Stamp(function="a_function", parameters={"a": 1, "b": "hey"}),
        )
    )

    assert len(history) == 2

    history2 = history.add({"message": "hey", "versions": {"some_package": "1.2.3"}})
    assert len(history2) == 3
    assert str(history) != history.pretty()

    # Ensure that we can index by int, str or datetime.
    assert (
        history[1]
        == history[history[1].timestamp]
        == history[history[1].timestamp.isoformat()]
    )

    with pytest.raises(KeyError):
        history["not_a_key"]

    with pytest.raises(KeyError):
        history[datetime.now()]

    with pytest.raises(KeyError):
        history[(1, 2)]


def test_bad_stamp_init():
    with pytest.raises(
        ValueError, match="History record must have a message or a function"
    ):
        Stamp()


def test_str_and_pretty():
    s = Stamp(message="dummy")
    assert str(s) != s.pretty()


def test_from_yaml_roundtrip():
    s = Stamp(message="dummy", parameters={"a": 1, "b": "hey"})
    xx = repr(s)

    s2 = Stamp.from_repr(xx)

    assert s2 == s
