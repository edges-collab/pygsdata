"""Functions for working with earth/sky coordinates."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytz
from astropy import coordinates as apc
from astropy import time as apt
from astropy import units as apu

from . import constants as const


def utc2lst(utc_times, longitude):
    """
    Convert an array representing UTC date/time to a 1D array of LST date/time.

    Parameters
    ----------
    utc_times : array-like
        Nx6 array of floats or integers, where each row is of the form
        [yyyy, mm, dd,HH, MM, SS]. It can also be a 6-element 1D array.
    longitude : float
        Terrestrial longitude of observatory (float) in degrees.

    Returns
    -------
    LST : 1D array of LST dates/times

    Examples
    --------
    >>> LST = utc2lst(utc_times, -27.5)
    """
    # convert input array to "int"
    if not isinstance(utc_times[0], dt.datetime):
        utc_times = [
            dt.datetime(*utc, tzinfo=pytz.utc)
            for utc in np.atleast_2d(utc_times).astype(int)
        ]

    # python "datetime" to astropy "Time" format
    t = apt.Time(utc_times, format="datetime", scale="utc")

    # necessary approximation to compute sidereal time
    t.delta_ut1_utc = 0

    return t.sidereal_time("apparent", str(longitude) + "d", model="IAU2006A").value


def moon_azel(times: apt.Time, obs_location: apc.EarthLocation) -> np.ndarray:
    """Get local coordinates of the Sun using Astropy."""
    moon = apc.get_moon(times).transform_to(apc.AltAz(location=obs_location))
    return moon.az.deg, moon.alt.deg


def sun_azel(times: apt.Time, obs_location: apc.EarthLocation) -> np.ndarray:
    """Get local coordinates of the Sun using Astropy."""
    sun = apc.get_moon(times).transform_to(apc.AltAz(location=obs_location))
    return sun.az.deg, sun.alt.deg


def f2z(fe: float | np.ndarray) -> float | np.ndarray:
    """Convert observed 21cm frequency to redshift."""
    # Constants and definitions
    c = 299792458  # wikipedia, m/s
    f21 = 1420.40575177e6  # wikipedia,
    lambda21 = c / f21  # frequency to wavelength, as emitted
    # frequency to wavelength, observed. fe comes in MHz but it
    # has to be converted to Hertz
    lmbda = c / (fe * 1e6)
    return (lmbda - lambda21) / lambda21


def z2f(z: float | np.ndarray) -> float | np.ndarray:
    """Convert observed redshift to 21cm frequency."""
    # Constants and definitions
    c = 299792458  # wikipedia, m/s
    f21 = 1420.40575177e6  # wikipedia,
    l21 = c / f21  # frequency to wavelength, as emitted
    lmbda = l21 * (1 + z)
    return c / (lmbda * 1e6)


def lst2gha(lst: float | np.ndarray) -> float | np.ndarray:
    """Convert LST to GHA."""
    gha = lst - const.galactic_centre_lst
    return gha % 24


def gha2lst(gha: float | np.ndarray) -> float | np.ndarray:
    """Convert GHA to LST."""
    lst = gha + const.galactic_centre_lst
    return lst % 24


def get_jd(d: dt.datetime) -> int:
    """Get the day of the year from a datetime object."""
    dt0 = dt.datetime(d.year, 1, 1, tzinfo=d.tzinfo)
    return (d - dt0).days + 1


def dt_from_jd(y: int, d: int, *args, tzinfo=pytz.utc) -> dt.datetime:
    """Get a datetime object from a julian date."""
    begin = dt.datetime(y, 1, 1, *args, tzinfo=tzinfo)
    return begin + dt.timedelta(days=d - 1)


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
