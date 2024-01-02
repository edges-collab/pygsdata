"""Selection functions for GSData objects."""
from __future__ import annotations

import logging
import numpy as np
import warnings
from astropy import units as un
from astropy.coordinates import Longitude
from astropy.time import Time
from typing import Tuple, Union

from .gsdata import GSData
from .register import gsregister

logger = logging.getLogger(__name__)

FreqType = un.Quantity[un.MHz]
FreqRangeType = tuple[FreqType, FreqType]
LSTType = Union[un.Quantity[un.hourangle], Longitude]
LSTRangeType = tuple[LSTType, LSTType]


@gsregister("reduce")
def select_freqs(
    data: GSData,
    *,
    freq_range: FreqRangeType | None = None,
    indx: np.ndarray | None = None,
    **kwargs,
) -> GSData:
    """Select a subset of the frequency channels."""
    if "range" in kwargs:
        warnings.warn("The 'range' keyword is deprecated, use 'freq_range' instead.")
        freq_range = kwargs.pop("range")

    if kwargs:
        raise ValueError(f"Unknown keyword arguments: {kwargs}")

    mask = None
    if freq_range is not None:
        if not isinstance(freq_range[0], un.Quantity):
            logger.warning("frequency range given without units, assuming MHz.")
            freq_range = (freq_range[0] * un.MHz, freq_range[1] * un.MHz)

        mask = (data.freq_array >= freq_range[0]) & (data.freq_array <= freq_range[1])

    if indx is not None:
        if mask is None:
            mask = np.zeros(len(data.freq_array), dtype=bool)
        mask[indx] = True

    if mask is None:
        return data

    return data.update(
        data=data.data[..., mask],
        freq_array=data.freq_array[mask],
        nsamples=data.nsamples[..., mask],
        flags={k: v.select(idx=mask, axis="freq") for k, v in data.flags.items()},
    )


def _mask_times(data: GSData, mask: np.ndarray) -> GSData:
    if mask is None:
        return data

    return data.update(
        data=data.data[:, :, mask],
        time_array=data.time_array[mask],
        time_ranges=data.time_ranges[mask],
        auxiliary_measurements={
            k: v[mask] for k, v in data.auxiliary_measurements.items()
        },
        nsamples=data.nsamples[:, :, mask],
        flags={k: v.select(idx=mask, axis="time") for k, v in data.flags.items()},
        residuals=data.residuals[:, :, mask] if data.residuals is not None else None,
    )


@gsregister("reduce")
def select_times(
    data: GSData,
    *,
    time_range: tuple[Time | float, Time | float] | None = None,
    fmt: str = "jd",
    indx: np.ndarray | None = None,
    load: int | str = "ant",
    **kwargs,
) -> GSData:
    """Select a subset of the times."""
    if data.in_lst:
        raise ValueError("LST-binned data cannot be selected on times.")

    if "range" in kwargs:
        warnings.warn("The 'range' keyword is deprecated, use 'time_range' instead.")
        time_range = kwargs.pop("range")

    if kwargs:
        raise ValueError(f"Unknown keyword arguments: {kwargs}")

    if isinstance(load, str):
        load = data.loads.index(load)

    mask = None
    if time_range is not None:
        if len(time_range) != 2:
            raise ValueError("range must be a length-2 tuple")

        if not isinstance(time_range[0], Time):
            time_range = (
                Time(time_range[0], format=fmt),
                Time(time_range[1], format=fmt),
            )

            t = data.time_array[:, load]
            mask = (t >= time_range[0]) & (t <= time_range[1])

    if indx is not None:
        if mask is None:
            mask = np.zeros((len(data.time_array),), dtype=bool)
        mask[indx] = True

    return _mask_times(data, mask)


@gsregister("reduce")
def select_lsts(
    data: GSData,
    *,
    lst_range: LSTRangeType | None = None,
    indx: np.ndarray | None = None,
    load: int | str = "ant",
    gha: bool = False,
    use_alan_coordinates: bool = False,
    **kwargs,
) -> GSData:
    """Select a subset of the times."""
    if kwargs:
        raise ValueError(f"Unknown keyword arguments: {kwargs}")

    if load == "all":
        load = slice(None)
    if isinstance(load, str):
        load = data.loads.index(load)

    mask = None
    if lst_range is not None:
        if len(lst_range) != 2:
            raise ValueError("range must be a length-2 tuple")

        if not isinstance(lst_range[0], Longitude):
            lst_range = (
                lst_range[0] % 24 * un.hourangle,
                lst_range[1] % 24 * un.hourangle,
            )

        t = data.gha[:, load] if gha else data.lst_array[:, load]

        # In case we have negative LST/GHA
        t = t % (24 * un.hourangle)

        if lst_range[0] > lst_range[1]:
            mask = (t >= lst_range[1]) & (t <= lst_range[0])
        else:
            mask = (t >= lst_range[0]) & (t <= lst_range[1])

        # Account for the case of load=='all' -- in this case require all loads
        # to be within the range.
        if mask.ndim == 2:
            mask = np.all(mask, axis=1)

    if indx is not None:
        if mask is None:
            mask = np.ones((len(data.time_array),), dtype=bool)

        if indx.dtype == bool:
            mask[~indx] = False
        else:
            for i in np.arange(data.ntimes):
                if i not in indx:
                    mask[i] = False

    return _mask_times(data, mask)


@gsregister("reduce")
def prune_flagged_integrations(data: GSData, **kwargs) -> GSData:
    """Remove integrations that are flagged for all freq-pol-loads."""
    flg = np.all(data.complete_flags, axis=(0, 1, 3))
    return _mask_times(data, ~flg)
