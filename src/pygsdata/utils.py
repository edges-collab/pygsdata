"""Utility functions."""

from collections.abc import Sequence

import h5py
import numpy as np
from astropy import units as un
from astropy.coordinates import Angle
from astropy.time import Time

# Target chunk size for checksummed HDF5 datasets. Kept at the size of HDF5's default
# chunk cache (1 MiB) so that partial reads of a chunk can be served from the cache.
_CHUNK_TARGET_BYTES = 1024**2


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


def calculate_rms(array: np.ndarray, digits=3, **kwargs):
    """Compute RMS of an array and round to the given number of decimal digits.

    Parameters
    ----------
    array
        Input array (array-like).
    digits
        Number of decimal places for the result.

    Returns
    -------
    float
        sqrt(mean(array**2)) rounded to `digits` decimals.
    """
    rms = np.sqrt(np.nanmean(array**2, **kwargs))
    return np.round(rms, digits)


def chunk_shape(
    shape: tuple[int, ...], itemsize: int, target: int = _CHUNK_TARGET_BYTES
) -> tuple[int, ...]:
    """Compute an HDF5 chunk shape of roughly ``target`` bytes.

    Trailing axes are kept whole for as long as they fit, so that a chunk of a
    (load, pol, time, freq) array spans full spectra. The first axis that does not
    fit is split, and all axes before it get a chunk size of one.
    """
    chunks = [1] * len(shape)
    nbytes = itemsize
    for i in range(len(shape) - 1, -1, -1):
        if nbytes * shape[i] <= target:
            chunks[i] = shape[i]
            nbytes *= shape[i]
        else:
            chunks[i] = max(1, target // nbytes)
            break
    return tuple(chunks)


def write_h5_dataset(
    grp: h5py.Group, name: str, value, checksum: bool = True
) -> h5py.Dataset:
    """Write a dataset to an HDF5 group, with a Fletcher32 checksum if possible.

    Checksums require chunked storage, so they are only applied to numeric arrays
    with at least one dimension and non-zero size. Everything else is written as a
    plain dataset.
    """
    arr = np.asarray(value)
    if checksum and arr.ndim > 0 and arr.size > 0 and arr.dtype.kind in "biufc":
        return grp.create_dataset(
            name,
            data=arr,
            chunks=chunk_shape(arr.shape, arr.dtype.itemsize),
            fletcher32=True,
        )

    grp[name] = value
    return grp[name]


def find_unreadable_datasets(grp: h5py.Group) -> list[str]:
    """Return the names of all datasets in ``grp`` that fail to read.

    For datasets written with a Fletcher32 checksum, a read failure means that the
    checksum did not match the stored data, i.e. the data is corrupted.
    """
    bad = []

    def _check(name, obj):
        if isinstance(obj, h5py.Dataset):
            try:
                obj[()]
            except OSError:
                bad.append(obj.name)

    grp.visititems(_check)
    return bad
