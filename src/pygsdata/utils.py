"""Utility functions."""

from collections.abc import Sequence

import numpy as np
from astropy import units as un
from astropy.coordinates import Angle
from astropy.time import Time

from edges import modeling as mdl
from .gsdata import GSData


def time_concat(arrays: Sequence[Time], axis: int = 0) -> Time:
    """Concatenate Time objects along axis.

    This is required because simple np.concatenate returns a numpy array of "objects"
    instead of a new Time object.
    """
    data = np.concatenate([a.jd for a in arrays], axis=axis)
    return Time(data, format="jd", scale=arrays[0].scale, location=arrays[0].location)


def angle_centre(a: Angle, b: Angle, p: float = 0.5):
    """Find the central point between two angles.

    This takes care of the cyclical nature of angles by
    enforcing that a < b.

    Parameters
    ----------
    a
        Angle(s) defining the lower bound
    b
        Angle(s) defining the upper bound
    p
        The fractional distance between a and b to return.

    Examples
    --------
    Let's go::

    >>> from astropy import units as u, Angle
    >>> angle_centre(Angle(0*un.hourangle), Angle(1*un.hourangle))
    >>> 0.5 hourangle
    >>> angle_centre(Angle(2*un.hourangle), Angle(0*un.hourangle))
    >>> 13 hourangle
    >>> angle_centre(Angle(23*un.hourangle), Angle(1*un.hourangle))
    >>> 0 hourangle
    >>> angle_centre(Angle(0*un.hourangle), Angle(1*un.hourangle), p=0.75)
    >>> 0.75 hourangle
    """
    kls = type(a)  # could be Angle or Longitude/Latitude
    if a.shape != b.shape:
        raise ValueError(f"a and b must have same shape, got a={a.shape}, b={b.shape}")

    ahr = a.hourangle
    bhr = b.hourangle

    if p < 0 or p > 1:
        raise ValueError("p must be between 0 and 1")

    if np.isscalar(bhr):
        if bhr < ahr:
            bhr += 24.0
    else:
        bhr[bhr < ahr] += 24.0

    return kls((ahr * (1 - p) + bhr * p) << un.hourangle)


def get_thermal_noise(data: GSData, n_terms=20):
    thermal_noise = []

    for i in range(len(data.lsts)):
        model = mdl.LinLog(n_terms=n_terms)
        model_fit_freqs = data.freqs

        LL = model.at(x=model_fit_freqs)
        res = LL.fit(ydata=data.data[0, 0, i, :], xdata=model_fit_freqs)

        thermal_noise.append(calculate_rms(res.residual))

    thermal_noise = np.array(thermal_noise)

    return thermal_noise


def calculate_rms(array, digits=3):
    rms = np.sqrt(np.mean(array**2))
    return round(rms, digits)