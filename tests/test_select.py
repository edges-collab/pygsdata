"""Tests of the selectors."""

import numpy as np
import pytest
from astropy import units as un

from pygsdata import GSData, GSFlag, select


def test_select_loads(power_gsdata):
    """Test selecting loads."""
    gsd = select.select_loads(power_gsdata, loads=("ant",))
    assert gsd.nloads == 1

    gsd1 = select.select_loads(power_gsdata, indx=[0])
    assert gsd1.nloads == 1


class TestSelectFreqs:
    """Test selecting frequencies."""

    def test_using_freqrange(self, simple_gsdata: GSData):
        """Test that using freq_range works as expected."""
        new = select.select_freqs(simple_gsdata, freq_range=(60 * un.MHz, 80 * un.MHz))

        assert new.freqs.min() >= 60 * un.MHz
        assert new.freqs.max() <= 80 * un.MHz
        assert new.data.shape != simple_gsdata.data.shape

    def test_using_freqrange_without_units(self, simple_gsdata: GSData):
        """Test that using freq_range without units raises a warning but works."""
        with pytest.warns(
            UserWarning, match="frequency range given without units, assuming MHz"
        ):
            new = select.select_freqs(simple_gsdata, freq_range=(60, 80))

        assert new.freqs.min() >= 60 * un.MHz
        assert new.freqs.max() <= 80 * un.MHz
        assert new.data.shape != simple_gsdata.data.shape

    def test_using_indx(self, simple_gsdata: GSData):
        """Test that passing a slice index works."""
        new = select.select_freqs(simple_gsdata, indx=slice(None, None, 2))

        assert new.freqs.size == simple_gsdata.freqs.size // 2

    def test_using_both(self, simple_gsdata: GSData):
        """Test that passing both index and freq_range OR together."""
        new = select.select_freqs(
            simple_gsdata, indx=[0, 25], freq_range=(60 * un.MHz, 80 * un.MHz)
        )

        assert new.freqs.min() == 50 * un.MHz
        assert new.freqs.max() <= 80 * un.MHz
        assert new.nfreqs < simple_gsdata.nfreqs


class TestSelectTimes:
    """Tests of the select_times functionality."""

    def test_integer_load(self, power_gsdata: GSData):
        """Test that passing an integer for load works as expected."""
        new1 = select.select_times(
            power_gsdata,
            load=0,
            time_range=(power_gsdata.times[0, 1], power_gsdata.times[0, -1]),
        )
        new2 = select.select_times(
            power_gsdata,
            load="ant",
            time_range=(power_gsdata.times[0, 1], power_gsdata.times[0, -1]),
        )

        new3 = select.select_times(
            power_gsdata,
            load="internal_load",
            time_range=(power_gsdata.times[0, 1], power_gsdata.times[0, -1]),
        )

        assert new1 == new2
        assert new1 != new3

    def test_bad_time_range(self, simple_gsdata: GSData):
        """Test that error is raised for non-length-2 time_range."""
        with pytest.raises(ValueError, match="range must be a length-2 tuple"):
            select.select_times(simple_gsdata, time_range=(1, 2, 3))

    def test_time_range_as_time_object(self, simple_gsdata: GSData):
        """Test that passing time_range as Time objects works."""
        new = select.select_times(
            simple_gsdata, time_range=(simple_gsdata.times[0], simple_gsdata.times[1])
        )
        assert new.ntimes == 2

    def test_time_range_as_float(self, simple_gsdata: GSData):
        """Test that passing simple floats with implicit format jd works."""
        new = select.select_times(
            simple_gsdata,
            time_range=(simple_gsdata.times[0].jd, simple_gsdata.times[1].jd),
        )
        assert new.ntimes == 2

    def test_using_indx(self, simple_gsdata: GSData):
        """Test that passing simple indx works."""
        new = select.select_times(simple_gsdata, indx=[0, 3])
        assert new.ntimes == 2

    def test_using_both(self, simple_gsdata: GSData):
        """Test that passing both indx and time_range ORs them together."""
        new = select.select_times(
            simple_gsdata,
            indx=[0],
            time_range=(simple_gsdata.times[1], simple_gsdata.times[-1]),
        )
        # Since indx and time_range are ORed together, we still have all the times.
        assert new.data.shape == simple_gsdata.data.shape


class TestSelectLSTs:
    """Tests of the select_lsts functionality."""

    def test_integer_load(self, power_gsdata: GSData):
        """Test that passing an integer load works."""
        lsts = power_gsdata.lsts[:, 1]
        new1 = select.select_lsts(
            power_gsdata, load=0, lst_range=(lsts.min(), lsts.max())
        )
        new2 = select.select_lsts(
            power_gsdata, load="ant", lst_range=(lsts.min(), lsts.max())
        )
        new3 = select.select_lsts(
            power_gsdata, load="internal_load", lst_range=(lsts.min(), lsts.max())
        )

        assert new1.data.shape == new2.data.shape
        assert new1.data.shape != new3.data.shape

    def test_bad_range(self, simple_gsdata: GSData):
        """Test that passing a length-3 lst_range raises an error."""
        with pytest.raises(ValueError, match="range must be a length-2 tuple"):
            select.select_lsts(simple_gsdata, lst_range=(1, 2, 3))

    def test_lst_range_as_longitude(self, simple_gsdata: GSData):
        """Test that passing an astropy Angle or Longitude works for lst_range."""
        new = select.select_lsts(
            simple_gsdata,
            lst_range=(simple_gsdata.lsts[0, 0], simple_gsdata.lsts[1, 0]),
        )
        assert new.ntimes == 2

    def test_lst_range_as_float(self, simple_gsdata: GSData):
        """Test that passing simple floats as lst_range works, interpreted as hours."""
        new = select.select_lsts(
            simple_gsdata,
            lst_range=(simple_gsdata.lsts[0, 0].hour, simple_gsdata.lsts[1, 0].hour),
        )
        assert new.ntimes == 2

    def test_using_indx(self, simple_gsdata: GSData):
        """Test that passing indx works."""
        new = select.select_lsts(simple_gsdata, indx=[0, 3])
        assert new.ntimes == 2

    def test_using_both(self, simple_gsdata: GSData):
        """Test that using both indx and lst_range ORs them together."""
        new = select.select_lsts(
            simple_gsdata,
            indx=[0],
            lst_range=(simple_gsdata.lsts[1, 0], simple_gsdata.lsts[-1, 0]),
        )
        # Since indx and time_range are ORed together, we still have all the times.
        assert new.data.shape == simple_gsdata.data.shape

    def test_gha(self, simple_gsdata: GSData):
        """Test that setting the gha=True flag works."""
        new = select.select_lsts(simple_gsdata, gha=True, lst_range=(0, 24))
        assert new.ntimes == simple_gsdata.ntimes

    def test_load_all(self, power_gsdata: GSData):
        """Test that setting load='all' tests all loads for their lsts."""
        new = select.select_lsts(
            power_gsdata,
            lst_range=(power_gsdata.lsts[0, 0], power_gsdata.lsts[-1, 0]),
            load="all",
        )
        # The last time is outside the range because one of the loads is outside.
        assert new.ntimes == power_gsdata.ntimes - 1


class TestSelectLoads:
    """Tests of the select_loads functionality."""

    def test_specify_loads(self, power_gsdata: GSData):
        """Test that specifying loads directly works."""
        new = select.select_loads(power_gsdata, loads=power_gsdata.loads[:2])
        assert new.nloads == 2

    def test_specify_indx(self, power_gsdata: GSData):
        """Test that specifying the indx works."""
        new = select.select_loads(power_gsdata, indx=[0, 2])
        assert new.nloads == 2
        assert new.loads == power_gsdata.loads[::2]

    def test_specify_indx_array(self, power_gsdata: GSData):
        """Test that specifying the indx as an array works."""
        new = select.select_loads(power_gsdata, indx=np.array([0, 2]))
        assert new.nloads == 2
        assert new.loads == power_gsdata.loads[::2]

    def test_specify_both(self, power_gsdata: GSData):
        """Test that specifying both indx and loads ORs them together."""
        new = select.select_loads(
            power_gsdata, indx=[0], loads=[power_gsdata.loads[-1]]
        )
        assert new.nloads == 2
        assert new.loads == power_gsdata.loads[::2]


class TestPruneFlaggedIntegrations:
    """Tests of the prune_flagged_integrations functionality."""

    def test_prune_none(self, simple_gsdata: GSData):
        """Test that pruning on a zero-flag dataset doesn't remove any data."""
        new = select.prune_flagged_integrations(simple_gsdata)
        assert new.data.shape == simple_gsdata.data.shape

    def test_prune_one(self, simple_gsdata: GSData):
        """Testing pruning a single integration."""
        flags = np.zeros(simple_gsdata.ntimes, dtype=bool)
        flags[0] = True

        new = simple_gsdata.add_flags("simple", GSFlag(flags=flags, axes=("time",)))
        new = select.prune_flagged_integrations(new)
        assert new.ntimes == simple_gsdata.ntimes - 1

    def test_prune_all(self, simple_gsdata: GSData):
        """Test pruning ALL integrations."""
        flags = np.ones(simple_gsdata.ntimes, dtype=bool)

        new = simple_gsdata.add_flags("simple", GSFlag(flags=flags, axes=("time",)))
        new = select.prune_flagged_integrations(new)
        assert new.ntimes == 0
