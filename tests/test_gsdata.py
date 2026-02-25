"""Tests of the GSData object."""

import pickle
import re
from datetime import timedelta
from pathlib import Path

import attrs
import numpy as np
import pytest
from astropy import units as un
from astropy.coordinates import Longitude
from astropy.time import Time

from pygsdata import (
    GSData,
    GSFlag,
    Stamp,
    select_freqs,
    select_times,
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

    with pytest.raises(TypeError, match="flags values must be GSFlag instances"):
        simple_gsdata.update(
            flags={"new": np.zeros(simple_gsdata.data.shape, dtype=bool)}
        )

    with pytest.raises(ValueError, match="residuals must have the same shape as data"):
        simple_gsdata.update(residuals=np.zeros((2, 1, 50, 100)))

    with pytest.raises(
        ValueError, match=re.escape("times must have the size (ntimes, nloads)")
    ):
        simple_gsdata.update(times=simple_gsdata.times[:-1])

    with pytest.raises(
        ValueError,
        match=re.escape("time_ranges must have the size (ntimes, nloads, 2)"),
    ):
        simple_gsdata.update(time_ranges=simple_gsdata.time_ranges[:-1])

    with pytest.raises(
        ValueError, match=re.escape("time_ranges must all be greater than zero")
    ):
        simple_gsdata.update(time_ranges=simple_gsdata.time_ranges[..., ::-1])

    with pytest.raises(
        ValueError,
        match=re.escape("effective_integration_time must be greater than zero"),
    ):
        simple_gsdata.update(effective_integration_time=-1 * un.s)

    with pytest.raises(
        ValueError, match=re.escape("effective_integration_time must be a scalar")
    ):
        simple_gsdata.update(effective_integration_time=np.ones((50, 273, 3)) * un.s)

    with pytest.raises(
        ValueError, match=re.escape("lsts must have the size (ntimes, nloads)")
    ):
        simple_gsdata.update(lsts=simple_gsdata.lsts[:-1])

    with pytest.raises(
        ValueError, match=re.escape("lst_ranges must have the size (ntimes, nloads, 2)")
    ):
        simple_gsdata.update(lst_ranges=simple_gsdata.lst_ranges[:-1])


def test_read_bad_filetype():
    with pytest.raises(ValueError, match="Unrecognized file type"):
        GSData.from_file("a_bad_file.txt")


def test_read_bad_selector(simple_gsdata, tmp_path):
    pth = tmp_path / "test.pkl"
    with Path(pth).open("wb") as fl:
        pickle.dump(simple_gsdata, fl)

    with pytest.raises(ValueError, match="Unrecognized selector"):
        GSData.from_file(
            pth,
            reader="gspkl",
            selectors={"bad_selector": {"some_key": 123}},
        )


def test_aux_none_is_ok(simple_gsdata):
    new = simple_gsdata.update(auxiliary_measurements=None)
    assert new.auxiliary_measurements is None


def test_effective_integration_time_array(simple_gsdata):
    new = simple_gsdata.update(
        effective_integration_time=simple_gsdata.effective_integration_time
    )
    assert np.all(
        new.effective_integration_time == simple_gsdata.effective_integration_time
    )


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
        no_flags.add_flags("new", np.zeros((1, 2, 3, 4)))

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
    assert simple_gsdata.get_initial_yearday() == "2022:320"

    with pytest.raises(ValueError, match="Cannot return minutes without hours"):
        simple_gsdata.get_initial_yearday(minutes=True)


def test_iterators(simple_gsdata):
    for slc in simple_gsdata.time_iter():
        assert simple_gsdata.data[slc].shape == simple_gsdata.data[:, :, 0].shape

    for slc in simple_gsdata.load_iter():
        assert simple_gsdata.data[slc].shape == simple_gsdata.data[0].shape

    for slc in simple_gsdata.freq_iter():
        assert simple_gsdata.data[slc].shape == simple_gsdata.data[:, :, :, 0].shape


def test_trivial_model(simple_gsdata):
    new = simple_gsdata.update(residuals=np.zeros_like(simple_gsdata.data))
    assert np.all(new.residuals == 0)
    assert np.array_equal(new.model, simple_gsdata.data)

    assert simple_gsdata.model is None


def test_partial_io_gspkl(simple_gsdata, tmp_path):
    pth = tmp_path / "test.pkl"
    with Path(pth).open("wb") as fl:
        pickle.dump(simple_gsdata, fl)

    new = GSData.from_file(
        pth,
        reader="gspkl",
        selectors={
            "time_selector": {"indx": np.arange(0, 10, 2)},
            "lst_selector": {"lst_range": (0, 24)},
            "freq_selector": {"freq_range": (50 * un.MHz, 70 * un.MHz)},
            "load_selector": {"loads": ("ant",)},
        },
    )

    assert new != simple_gsdata


def test_multifile_read(simple_gsdata, tmp_path):
    part1 = select_times(simple_gsdata, indx=list(range(5)))
    part2 = select_times(simple_gsdata, indx=list(range(5, 10)))

    part1.write_gsh5(tmp_path / "part1.gsh5")
    part2.write_gsh5(tmp_path / "part2.gsh5")

    new = GSData.from_file(
        [tmp_path / "part1.gsh5", tmp_path / "part2.gsh5"], concat_axis="time"
    )
    assert np.all(new.data == simple_gsdata.data)


def test_add_errors(
    simple_gsdata, simple_gsdata_noaux, flagged_gsdata, modelled_gsdata
):
    new = simple_gsdata + simple_gsdata
    assert np.all(new.data == simple_gsdata.data * 2)

    with pytest.raises(
        ValueError, match="Cannot add GSData objects with different shapes"
    ):
        simple_gsdata + select_freqs(
            simple_gsdata, freq_range=(50 * un.MHz, 70 * un.MHz)
        )

    new = simple_gsdata + simple_gsdata_noaux
    assert new.auxiliary_measurements is not None

    new = simple_gsdata_noaux + simple_gsdata
    assert new.auxiliary_measurements is not None

    flagged_gsdata = flagged_gsdata.update(
        auxiliary_measurements={
            k: flagged_gsdata.auxiliary_measurements[k] + 3
            for k in flagged_gsdata.auxiliary_measurements.columns
        }
    )
    with pytest.warns(UserWarning, match="Overlapping auxiliary measurements exist"):
        new = simple_gsdata + flagged_gsdata

    assert new.auxiliary_measurements is not None

    with pytest.raises(
        ValueError, match="Cannot add GSData objects with different times"
    ):
        simple_gsdata + simple_gsdata.update(times=simple_gsdata.times + 1 * un.day)

    modelled = modelled_gsdata + modelled_gsdata
    assert np.all(modelled.residuals == 0)
    assert np.all(modelled.data == simple_gsdata.data * 2)


def test_gha(simple_gsdata):
    assert simple_gsdata.gha.shape == simple_gsdata.times.shape
    assert isinstance(simple_gsdata.gha, Longitude)


def test_get_moon_sun(simple_gsdata):
    az, el = simple_gsdata.get_moon_azel()
    assert az.size == el.size == simple_gsdata.times.size

    az, el = simple_gsdata.get_sun_azel()
    assert az.size == el.size == simple_gsdata.times.size


@pytest.mark.parametrize("which_flags", [None, (), "all", "twice"])
@pytest.mark.parametrize("ignore_flags", [(), "all"])
@pytest.mark.parametrize("data", ["simple_gsdata", "flagged_gsdata"])
def test_cumulative_flags_cases(data, which_flags, request, ignore_flags):
    data = request.getfixturevalue(data)

    if which_flags == "all":
        which_flags = tuple(data.flags.keys())
    elif which_flags == "twice":
        which_flags = 2 * tuple(data.flags.keys())

    if ignore_flags == "all":
        ignore_flags = tuple(data.flags.keys())

    flags = data.get_cumulative_flags(
        which_flags=which_flags, ignore_flags=ignore_flags
    )
    assert np.all(flags == 0)


def test_get_initial_yearday(simple_gsdata):
    yd = simple_gsdata.get_initial_yearday(hours=True, minutes=True)

    yd1 = simple_gsdata.get_initial_yearday(hours=True, minutes=False)
    assert yd.startswith(yd1)

    yd2 = simple_gsdata.get_initial_yearday(hours=False, minutes=False)
    assert yd1.startswith(yd2)

    with pytest.raises(ValueError, match="Cannot return minutes without hours"):
        simple_gsdata.get_initial_yearday(minutes=True)


def test_add_flags_from_file(simple_gsdata: GSData, tmp_path: Path):
    flags = GSFlag(flags=np.zeros(simple_gsdata.data.shape, dtype=bool))
    flags.write_gsflag(tmp_path / "tmp.gsflag")

    new = simple_gsdata.add_flags("fileflags", tmp_path / "tmp.gsflag")
    assert new.nflagging_ops == 1
