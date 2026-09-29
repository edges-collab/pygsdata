"""Test the history module."""

import warnings
from datetime import UTC, datetime, timedelta
from importlib.metadata import version

import hickle
import pytest
import yaml

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
        history[datetime.now() + timedelta(hours=1)]

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


def test_from_yaml_roundtrip_timezon():
    s = Stamp(message="dummy", timestamp=datetime.now(tz=UTC))
    xx = repr(s)

    s2 = Stamp.from_repr(xx)

    assert s2 == s


def test_default_constructor():
    """Test that unknown tags are loaded ok."""
    unknown_tag = "!ANewTag 3.0"
    thing = yaml.load(unknown_tag, Loader=yaml.FullLoader)
    assert thing == "!ANewTag: 3.0"


def test_constructing_history_from_non_stamps():
    """Test that constructing a history from non-stamps fails."""
    h = History()
    with pytest.raises(TypeError, match="stamp must be a Stamp or a dictionary"):
        h.add((3, 4))


def test_non_imported_constructor():
    txt = "!!python/name:non.imported.module"
    with pytest.warns(UserWarning, match="History was not readable"):
        History.from_repr(txt)


def test_non_yamlable_parameter():
    """Test that non-yamlable parameters are sanitized when creating a history."""
    s = Stamp(message="dummy", parameters={"a": 1, "b": lambda x: 3})
    h = History(stamps=(s,))

    h2 = History.from_repr(repr(h))

    assert h2.stamps[0].parameters["a"] == 1
    assert isinstance(h2.stamps[0].parameters["b"], str)


def _old_style_dict():
    """Return a stamp dict as written before qualname/description were added."""
    return {
        "message": "",
        "function": "a_function",
        "parameters": {"a": 1},
        "versions": {"numpy": "1.0"},
        "timestamp": datetime.now().isoformat(),
    }


def test_old_style_stamp_loads_with_defaults():
    s = Stamp.from_yaml_dict(_old_style_dict())
    assert s.qualname == ""
    assert s.description == ""


def test_old_style_stamp_setstate():
    """Unpickling (e.g. via hickle) an old stamp fills in the new fields."""
    s = Stamp.__new__(Stamp)
    s.__setstate__(_old_style_dict())
    assert s.function == "a_function"
    assert s.description == ""
    assert isinstance(s.timestamp, datetime)


def test_hickle_roundtrip_no_warning(tmp_path):
    h = History((Stamp(message="a", qualname="mod.f", description="Do it."),))
    hickle.dump(h, tmp_path / "h.h5", mode="w")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        h2 = hickle.load(tmp_path / "h.h5")
    assert h2[0].description == "Do it."
    assert not hasattr(h2[0], "item_index")


def test_unknown_stamp_fields_are_dropped():
    d = {**_old_style_dict(), "from_the_future": 3}
    with pytest.warns(UserWarning, match="unknown history fields"):
        s = Stamp.from_yaml_dict(d)
    assert not hasattr(s, "from_the_future")

    with pytest.warns(UserWarning, match="unknown history fields"):
        h = History.from_repr(yaml.dump([d]))
    assert len(h) == 1


def test_str_and_pretty_show_description():
    s = Stamp(function="f", qualname="mod.f", description="Do [a] thing.")
    assert "mod.f" in str(s)
    assert "Do [a] thing." in str(s)
    assert r"Do \[a] thing." in s.pretty()


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({}, "no qualname recorded"),
        ({"qualname": "__main__.f"}, "defined in a script"),
        ({"qualname": "not_a_module.f"}, "cannot import not_a_module.f"),
        ({"qualname": "pygsdata.register.nope"}, "cannot import"),
        (
            {"qualname": "pygsdata.register.add_flags", "versions": {}},
            "pygsdata version recorded as None",
        ),
        (
            {
                "qualname": "pygsdata.register.add_flags",
                "versions": {"pygsdata": "0.0.1"},
            },
            "pygsdata version recorded as 0.0.1",
        ),
    ],
)
def test_parameter_descriptions_unavailable(kwargs, reason):
    pytest.importorskip("docstring_parser")
    s = Stamp(function="f", parameters={"filt": "x"}, **kwargs)
    with pytest.raises(LookupError, match=reason):
        s.parameter_descriptions()

    # pretty() never fails, it just says why there are no descriptions.
    pretty = s.pretty(annotate=True)
    assert "no descriptions:" in pretty
    assert reason in pretty


def test_parameter_descriptions_matching_version():
    pytest.importorskip("docstring_parser")
    s = Stamp(
        function="add_flags",
        qualname="pygsdata.register.add_flags",
        parameters={"filt": "x"},
        versions={"pygsdata": version("pygsdata")},
    )
    # Only parameters that were recorded are described.
    assert s.parameter_descriptions() == {
        "filt": "The name under which to store the flags."
    }
    assert "no descriptions" not in History([s]).pretty(annotate=True)
