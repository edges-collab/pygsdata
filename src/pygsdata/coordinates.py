"""Functions for working with earth/sky coordinates."""

import numpy as np
from astropy import coordinates as apc
from astropy import time as apt
from astropy import units as apu

from . import constants as const
from .types import AngleType


def moon_azel(times: apt.Time, obs_location: apc.EarthLocation) -> np.ndarray:
    """Get local coordinates of the Sun using Astropy."""
    moon = apc.get_body("moon", time=times).transform_to(
        apc.AltAz(location=obs_location)
    )
    return moon.az.deg, moon.alt.deg


def sun_azel(times: apt.Time, obs_location: apc.EarthLocation) -> np.ndarray:
    """Get local coordinates of the Sun using Astropy."""
    sun = apc.get_body("sun", time=times).transform_to(apc.AltAz(location=obs_location))
    return sun.az.deg, sun.alt.deg


def lst2gha(lst: AngleType) -> AngleType:
    """Convert LST to GHA."""
    return lst.__class__((lst - const.galactic_centre_lst) % (24 * apu.hourangle))


def gha2lst(gha: AngleType) -> AngleType:
    """Convert GHA to LST."""
    return gha.__class__((gha + const.galactic_centre_lst) % (24 * apu.hourangle))


def lst_to_earth_time(time: apt.Time, location: apc.EarthLocation) -> float:
    """Return a factor to convert one second of earth-measured time to an LST."""
    time2 = time + apt.TimeDelta(1 * apu.s)
    lst = time.sidereal_time("apparent", longitude=location.lon)
    lst2 = time2.sidereal_time("apparent", longitude=location.lon)
    return (lst2.arcsecond - lst.arcsecond) / 15


def lsts_to_times(
    lsts: AngleType, ref_time: apt.Time, location: apc.EarthLocation
) -> apt.Time:
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
    ref_lst = ref_time.sidereal_time("apparent", longitude=location.lon)
    lst_per_sec = lst_to_earth_time(ref_time, location)
    lst_diff = lsts - ref_lst
    sec_diff = apt.TimeDelta(lst_diff.arcsecond / 15 / lst_per_sec, format="sec")
    return ref_time + sec_diff
