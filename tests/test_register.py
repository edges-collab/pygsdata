"""Test the register module."""

from importlib.metadata import version

import pytest

from pygsdata import gsregister
from pygsdata.gsdata import GSData
from pygsdata.register import add_flags


@gsregister("calibrate")
def bad_gsfunc(data: GSData) -> GSData:
    return 3


def test_bad_gsfunc_return(simple_gsdata):
    with pytest.raises(TypeError, match="bad_gsfunc returned <class 'int'>"):
        bad_gsfunc(simple_gsdata)


@gsregister("calibrate")
def documented_gsfunc(data: GSData, scale, *extra, offset=1.0, **kwargs) -> GSData:
    """Scale and offset the data.

    Longer description that should not be stored.

    Parameters
    ----------
    data
        The input data.
    scale
        Factor by which to [multiply] the data.
    offset
        Value added to the data
        after scaling.
    """
    return data


@gsregister("calibrate")
def undocumented_gsfunc(data: GSData) -> GSData:
    return data


def test_stamp_records_all_parameters(simple_gsdata):
    stamp = documented_gsfunc(simple_gsdata, 2.0, message="testing").history[-1]

    assert stamp.function == "documented_gsfunc"
    assert stamp.qualname == "test_register.documented_gsfunc"
    assert stamp.description == "Scale and offset the data."
    assert stamp.message == "testing"
    # Positional args and defaults are both recorded; empty *args are not.
    assert stamp.parameters == {"scale": 2.0, "offset": 1.0}

    new = documented_gsfunc(simple_gsdata, 2.0, 3, 4, offset=0, foo="bar")
    assert new.history[-1].parameters == {
        "scale": 2.0,
        "extra": [3, 4],
        "offset": 0,
        "foo": "bar",
    }


def test_stamp_without_docstring(simple_gsdata):
    stamp = undocumented_gsfunc(simple_gsdata).history[-1]
    assert stamp.description == ""
    assert stamp.parameters == {}


def test_stamp_records_owning_package_version(flagged_gsdata):
    new = add_flags(flagged_gsdata, "new", flagged_gsdata.flags["hello"])
    stamp = new.history[-1]
    assert stamp.qualname == "pygsdata.register.add_flags"
    assert stamp.versions["pygsdata"] == version("pygsdata")


def test_annotated_pretty(simple_gsdata):
    pytest.importorskip("docstring_parser")
    stamp = documented_gsfunc(simple_gsdata, 2.0).history[-1]

    assert stamp.parameter_descriptions() == {
        "scale": "Factor by which to [multiply] the data.",
        "offset": "Value added to the data after scaling.",
    }
    pretty = stamp.pretty(annotate=True)
    assert r"# Factor by which to \[multiply] the data." in pretty
    assert "# Value added to the data after scaling." in pretty
    assert "# " not in stamp.pretty()


def test_new_fields_survive_gsh5_roundtrip(simple_gsdata, tmp_path):
    new = documented_gsfunc(simple_gsdata, 2.0)
    new.write_gsh5(tmp_path / "test.gsh5")
    stamp = GSData.from_file(tmp_path / "test.gsh5").history[-1]

    assert stamp.qualname == "test_register.documented_gsfunc"
    assert stamp.description == "Scale and offset the data."
    assert stamp.parameters == {"scale": 2.0, "offset": 1.0}
