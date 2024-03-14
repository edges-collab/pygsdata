"""Tests of coordinate transformations."""

import numpy as np
import pytest
from astropy import coordinates as apc
from astropy import time as apt
from astropy import units as apu
from pygsdata.constants import KNOWN_TELESCOPES
from pygsdata.coordinates import lst2gha, lsts_to_times


def test_lsts_to_times():
    """Test the conversion of LSTs to times."""
    lsts = np.arange(0, 24, 0.5)
    ref_time = apt.Time("2020-01-01T00:00:00")
    times = lsts_to_times(
        lsts=lsts,
        ref_time=apt.Time("2020-01-01T00:00:00"),
        location=KNOWN_TELESCOPES["edges-low"].location,
    )

    for time in times:
        assert np.abs((ref_time - time).sec) < 24 * 60 * 60


@pytest.mark.parametrize(
    "lsts",
    [
        np.linspace(0, 24, 25) * apu.hourangle,
        np.linspace(0, 2 * np.pi, 25) * apu.rad,
        apc.Longitude(np.linspace(0, 24, 25), unit=apu.hourangle),
    ],
)
def test_lst2gha(lsts):
    """Test the conversion of LST to GHA."""
    gha = lst2gha(lsts)
    assert gha[0] == gha[-1]
