"""Functions for working with earth/sky coordinates."""

from __future__ import annotations

import datetime as dt

import numpy as np
from astropy import coordinates as apc
from astropy import time as apt
from astropy import units as apu


def moon_azel(times: apt.Time, obs_location: apc.EarthLocation) -> np.ndarray:
    """Get local coordinates of the Sun using Astropy."""
    moon = apc.get_moon(times).transform_to(apc.AltAz(location=obs_location))
    return moon.az.deg, moon.alt.deg


def sun_azel(times: apt.Time, obs_location: apc.EarthLocation) -> np.ndarray:
    """Get local coordinates of the Sun using Astropy."""
    sun = apc.get_moon(times).transform_to(apc.AltAz(location=obs_location))
    return sun.az.deg, sun.alt.deg


def lst_to_earth_time(time: apt.Time) -> float:
    """Return a factor to convert one second of earth-measured time to an LST."""
    time2 = time + dt.timedelta(seconds=1)
    lst = time.sidereal_time("apparent")
    lst2 = time2.sidereal_time("apparent")
    return (lst2.arcsecond - lst.arcsecond) / 15


def lsts_to_times(
    lsts: np.typing.ArrayLike, ref_time: apt.Time, location: apc.EarthLocation
) -> list[apt.Time]:
    """Convert a list of LSTs to local times at a particular location.

    The times are generated close to (surrounding) a particular time.

    Recall that any particular LST maps to some time on an infinite number of days.
    Pass `ref_time` to set the time around which the LSTs will map.

    Parameters
    ----------
    lsts
        A list/array of LSTs in hours.
    ref_time
        An astropy time (in UTC) giving the time around which to find the LSTs.
    location
        The location at which the LSTs are defined.
    """
    ref_time.location = location
    ref_lst = ref_time.sidereal_time("apparent")
    lst_per_sec = lst_to_earth_time(ref_time)
    times = []
    for this_lst in lsts:
        lst_diff = apc.Longitude(this_lst * apu.hour) - ref_lst
        sec_diff = apt.TimeDelta(lst_diff.arcsecond / 15 / lst_per_sec, format="sec")
        times.append(ref_time + sec_diff)
    return times
