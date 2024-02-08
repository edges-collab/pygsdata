import numpy as np
from astropy import time as apt

from pygsdata.constants import KNOWN_LOCATIONS
from pygsdata.coordinates import lsts_to_times


def test_lsts_to_times():
    lsts = np.arange(0, 24, 0.5)
    ref_time = apt.Time("2020-01-01T00:00:00")
    times = lsts_to_times(
        lsts=lsts,
        ref_time=apt.Time("2020-01-01T00:00:00"),
        location=KNOWN_LOCATIONS["edges"],
    )

    for time in times:
        assert np.abs((ref_time - time).sec) < 24 * 60 * 60
