"""Utility functions."""
from collections.abc import Sequence

import numpy as np
from astropy.time import Time


def time_concat(arrays: Sequence[Time], axis: int = 0) -> Time:
    """Concatenate Time objects along axis.

    This is required because simple np.concatenate returns a numpy array of "objects"
    instead of a new Time object.
    """
    data = np.concatenate([a.jd for a in arrays], axis=axis)
    return Time(data, format="jd", scale=arrays[0].scale, location=arrays[0].location)
