"""Test the concat module."""

import numpy as np
import pytest
from astropy import units as un
from mock_gsdata import flag_constant, mockgsd
from pygsdata.concat import concat


def test_concat_times():
    d1 = mockgsd(ntime=12)
    d2 = mockgsd(time0=(d1.times.max() + 40 * un.s).jd, ntime=10)

    d3 = concat([d1, d2], axis="time")
    assert d3.ntimes == 22
    assert d3.times[0] == d1.times[0]
    assert d3.times[-1] == d2.times[-1]


def test_concat_freqs():
    d1 = mockgsd(nfreq=12, freq_range=(50 * un.MHz, 100 * un.MHz))
    d2 = mockgsd(nfreq=10, freq_range=(100 * un.MHz, 150 * un.MHz))

    d3 = concat([d1, d2], axis="freq")
    assert d3.nfreqs == 22
    assert d3.freqs[0] == d1.freqs[0]
    assert d3.freqs[-1] == d2.freqs[-1]


def test_concat_loads():
    d1 = mockgsd(loads=("ant1",))
    d2 = mockgsd(loads=("ant2",))

    d3 = concat([d1, d2], axis="load")
    assert d3.nloads == 2
    assert d3.loads == ("ant1", "ant2")


def test_concat_with_flags():
    d1 = mockgsd(ntime=10, flag_creators={"flag1": flag_constant()})
    d2 = mockgsd(ntime=12, flag_creators={"flag1": flag_constant()})

    d3 = concat([d1, d2], axis="time")
    assert "flag1" in d3.flags
    assert d3.flags["flag1"].axes == ("time",)
    assert d3.flags["flag1"].flags.shape == (22,)


def test_concat_different_flags():
    d1 = mockgsd(flag_creators={"flag1": flag_constant()})
    d2 = mockgsd(flag_creators={"flag2": flag_constant()})

    with pytest.raises(ValueError, match="Flags must have the same keys"):
        concat([d1, d2], axis="time")


def test_concat_flags_without_concat_axis():
    d1 = mockgsd(flag_creators={"flag1": flag_constant(axes=("time",))})
    d2 = mockgsd(flag_creators={"flag1": flag_constant(axes=("time",))})

    with pytest.warns(UserWarning, match="Flags flag1 do not have a freq axis"):
        d3 = concat([d1, d2], axis="freq")

    assert "flag1" in d3.flags
    assert d3.flags["flag1"].axes == ("time",)
    assert np.all(d3.flags["flag1"] == d1.flags["flag1"])


def test_concat_bad_axis():
    d1 = mockgsd()
    d2 = mockgsd()

    with pytest.raises(ValueError, match="Axis must be"):
        concat([d1, d2], axis="badaxis")
