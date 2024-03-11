"""Tests of the GSData object."""

from datetime import datetime, timedelta

import attrs
import numpy as np
import pytest
from astropy import units as un
from astropy.time import Time
from pygsdata import (
    KNOWN_TELESCOPES,
    GSData,
    GSFlag,
    History,
    Stamp,
    Telescope,
    gsregister,
    select_freqs,
    select_lsts,
    select_times,
)


@pytest.fixture(scope="module")
def edges() -> Telescope:
    return KNOWN_TELESCOPES["edges-low"]


@pytest.fixture(scope="module")
def simple_gsdata(edges):
    return GSData(
        data=np.zeros((1, 1, 50, 100)),
        freqs=np.linspace(50, 100, 100) * un.MHz,
        times=Time(np.linspace(2459811, 2459812, 50)[:, np.newaxis], format="jd"),
        telescope=edges,
    )


@pytest.fixture(scope="module")
def power_gsdata(edges):
    return GSData(
        data=np.zeros((3, 1, 50, 100)),
        freqs=np.linspace(50, 100, 100) * un.MHz,
        times=Time(
            np.array(
                [
                    np.linspace(2459811, 2459812, 50),
                    np.linspace(2459811.001, 2459812.001, 50),
                    np.linspace(2459811.002, 2459812.002, 50),
                ]
            ).T,
            format="jd",
        ),
        telescope=edges,
    )


def test_default_power_load_names(power_gsdata):
    assert power_gsdata.loads == (
        "ant",
        "internal_load",
        "internal_load_plus_noise_source",
    )


def test_bad_load_names(edges):
    with pytest.raises(ValueError, match="If data has more than one source"):
        GSData(
            data=np.zeros((2, 1, 50, 100)),
            freqs=np.linspace(50, 100, 100) * un.MHz,
            times=Time(
                np.array(
                    [
                        np.linspace(2459811, 2459812, 50),
                        np.linspace(2459811.001, 2459812.001, 50),
                    ]
                ).T,
                format="jd",
            ),
            telescope=edges,
        )


@gsregister("calibrate")
def bad_gsfunc(data):
    return 3


def test_bad_gsfunc_return(simple_gsdata):
    with pytest.raises(TypeError, match="bad_gsfunc returned <class 'int'>"):
        bad_gsfunc(simple_gsdata)


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


def test_bad_gsdata_init(simple_gsdata: GSData):
    with pytest.raises(ValueError, match="data must have ndim in"):
        simple_gsdata.update(data=np.zeros((3, 4, 5)))

    with pytest.raises(ValueError, match="nsamples must have the same shape as data"):
        simple_gsdata.update(nsamples=np.zeros((2, 1, 50, 100)))

    with pytest.raises(TypeError, match="flags must be a dict"):
        simple_gsdata.update(flags=np.zeros(simple_gsdata.data.shape, dtype=bool))

    with pytest.raises(ValueError, match="flags keys must be strings"):
        simple_gsdata.update(
            flags={
                1: GSFlag(
                    flags=np.zeros(simple_gsdata.data.shape, dtype=bool),
                    axes=("load", "pol", "time", "freq"),
                )
            }
        )

    with pytest.raises(TypeError, match="freqs must be a Quantity"):
        simple_gsdata.update(freqs=np.linspace(50, 100, 100))

    with pytest.raises(ValueError, match="freqs must have units compatible with MHz"):
        simple_gsdata.update(freqs=np.linspace(50, 100, 100) * un.m)

    with pytest.raises(ValueError, match="freqs must have the size nfreqs"):
        simple_gsdata.update(freqs=simple_gsdata.freqs[:-1])

    with pytest.raises(ValueError, match="times must have ndim in "):
        simple_gsdata.update(times=simple_gsdata.times[:, 0])

    with pytest.raises(ValueError, match="loads must have the same length as"):
        simple_gsdata.update(loads=("ant", "another_one"))

    with pytest.raises(ValueError, match="loads must be a tuple of strings"):
        simple_gsdata.update(loads=(38,))

    with pytest.raises(
        ValueError,
        match="auxiliary_measurements must be length ntimes",
    ):
        simple_gsdata.update(auxiliary_measurements={"hey": np.linspace(50, 100, 75)})

    with pytest.raises(ValueError, match="data_unit must be one of"):
        simple_gsdata.update(data_unit="my_custom_string")


def test_read_bad_filetype():
    with pytest.raises(ValueError, match="Unrecognized file type"):
        GSData.from_file("a_bad_file.txt")


def test_update_history(simple_gsdata):
    assert len(simple_gsdata.history) == 0

    new = simple_gsdata.update(data=simple_gsdata.data + 1)
    assert len(new.history) == 0

    new = new.update(history=Stamp(message="just tagging a point in time"))
    assert len(new.history) == 1

    new = new.update(history={"function": "add"}, data=new.data + 1)
    assert len(new.history) == 2

    with pytest.raises(
        ValueError, match="History must be a Stamp object or dictionary"
    ):
        new.update(history="bad message")


def test_select_lsts_and_times(power_gsdata):
    indx = np.zeros(50, dtype=bool)
    indx[::2] = True

    lst = select_lsts(power_gsdata, indx=indx)
    tm = select_times(power_gsdata, indx=indx)

    assert lst == tm

    stime = select_times(power_gsdata, time_range=(2459811.5, 2459811.7))
    assert stime != power_gsdata


def test_select_lsts(power_gsdata: GSData):
    rng = (power_gsdata.lsts.min().hour, power_gsdata.lsts.max().hour)

    new = select_lsts(power_gsdata, lst_range=rng, load="all")
    # even though they're the same, the _file_appendable is switched off now
    assert new != power_gsdata
    assert np.allclose(new.data, power_gsdata.data)

    new = select_lsts(power_gsdata, lst_range=rng, load="ant")
    assert new != power_gsdata
    assert np.allclose(new.data, power_gsdata.data[0])

    with pytest.raises(ValueError, match="range must be a length-2 tuple"):
        select_lsts(power_gsdata, lst_range=[0, 2, 3])

    # Use different order of range
    new = select_lsts(power_gsdata, lst_range=(-2, 4))
    new2 = select_lsts(power_gsdata, lst_range=(22, 4))

    assert new == new2

    # Test with both indx and range
    new = select_lsts(power_gsdata, lst_range=rng, indx=np.arange(0, 50, 2))
    new2 = select_lsts(power_gsdata, indx=np.arange(0, 50, 2))

    assert new == new2


def test_select_freqs(simple_gsdata):
    new = select_freqs(simple_gsdata, freq_range=(50 * un.MHz, 70 * un.MHz))
    assert new.freqs.max() <= 70 * un.MHz


def test_add(simple_gsdata):
    with pytest.raises(TypeError, match="can only add GSData objects"):
        simple_gsdata + 3

    new_times = simple_gsdata.update(
        times=simple_gsdata.times + timedelta(days=1),
        time_ranges=simple_gsdata.time_ranges + 1 * un.day,
    )

    new_freqs = simple_gsdata.update(freqs=simple_gsdata.freqs + 50 * un.MHz)

    new_timefreq = new_times.update(freqs=new_freqs.freqs)

    with pytest.raises(
        ValueError,
        match="Cannot add GSData objects with different frequencies",
    ):
        simple_gsdata + new_timefreq

    doubled = simple_gsdata + simple_gsdata
    assert np.allclose(doubled.data, simple_gsdata.data * 2)


def test_moon_sun(simple_gsdata: GSData):
    az, el = simple_gsdata.get_moon_azel()
    assert len(az) == len(el) == simple_gsdata.ntimes


def test_cumulative_flags(simple_gsdata: GSData):
    # Add a few flags in...

    flg0 = np.zeros(simple_gsdata.data.shape, dtype=bool)
    no_flags = simple_gsdata.add_flags("zeros", flg0)

    flg1 = np.copy(flg0)
    flg1[:, :, 0, :] = True
    time_flags = no_flags.add_flags("time", flg1)

    flg2 = np.copy(flg0)
    flg2[:, :, :, 0] = True
    freq_flags = time_flags.add_flags("freq", flg2)

    assert np.array_equal(
        freq_flags.get_cumulative_flags(which_flags=("zeros",)), no_flags.complete_flags
    )

    assert np.array_equal(
        freq_flags.get_cumulative_flags(ignore_flags=("freq",)),
        time_flags.complete_flags,
    )

    assert np.array_equal(freq_flags.get_cumulative_flags(), freq_flags.complete_flags)

    with pytest.raises(ValueError, match="Flags for filter"):
        # can't add the same flags twice
        no_flags.add_flags("zeros", flg0)

    with pytest.raises(ValueError, match="Objects have different npols"):
        no_flags.add_flags("new", np.zeros((1, 2, 3)))

    with pytest.raises(ValueError, match="Cannot append to file without a filename"):
        no_flags.add_flags("time", flg1, append_to_file=True)

    new_no_flags = time_flags.remove_flags("time")

    flds = attrs.fields(GSData)
    for fld in flds:
        v1 = getattr(new_no_flags, fld.name)
        v2 = getattr(no_flags, fld.name)

        if not fld.eq:
            continue
        if fld.eq_key is not None:
            if fld.eq_key(v1) != fld.eq_key(v2):
                print(fld)
        elif v1 != v2:
            print(fld)

    assert new_no_flags == no_flags

    with pytest.raises(ValueError, match="No flags for filter"):
        no_flags.remove_flags("nonexistent")


def test_initial_yearday(simple_gsdata):
    assert simple_gsdata.get_initial_yearday() == "2022:231"

    with pytest.raises(ValueError, match="Cannot return minutes without hours"):
        simple_gsdata.get_initial_yearday(minutes=True)


def test_iterators(simple_gsdata):
    for slc in simple_gsdata.time_iter():
        assert simple_gsdata.data[slc].shape == simple_gsdata.data[:, :, 0].shape

    for slc in simple_gsdata.load_iter():
        assert simple_gsdata.data[slc].shape == simple_gsdata.data[0].shape

    for slc in simple_gsdata.freq_iter():
        assert simple_gsdata.data[slc].shape == simple_gsdata.data[:, :, :, 0].shape
