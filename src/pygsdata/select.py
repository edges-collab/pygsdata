"""Selection functions for GSData objects."""

from __future__ import annotations

import logging
import warnings
from typing import Union

import numpy as np
from astropy import units as un
from astropy.coordinates import Longitude
from astropy.time import Time

from .coordinates import lst2gha
from .gsdata import GSData
from .register import gsregister

logger = logging.getLogger(__name__)

FreqType = un.Quantity[un.MHz]
FreqRangeType = tuple[FreqType, FreqType]
LSTType = Union[un.Quantity[un.hourangle], Longitude]
LSTRangeType = tuple[LSTType, LSTType]


def freq_selector(
    _freqs: FreqType,
    *,
    freq_range: FreqRangeType | None = None,
    indx: np.ndarray | None = None,
) -> np.ndarray:
    """Select a subset of the frequency channels."""
    mask = None
    if freq_range is not None:
        if not isinstance(freq_range[0], un.Quantity):
            logger.warning("frequency range given without units, assuming MHz.")
            freq_range = (freq_range[0] * un.MHz, freq_range[1] * un.MHz)

        mask = (_freqs >= freq_range[0]) & (_freqs <= freq_range[1])

    if indx is not None:
        if mask is None:
            mask = np.zeros(len(_freqs), dtype=bool)
        mask[indx] = True

    if mask is None:
        return np.ones(len(_freqs), dtype=bool)
    else:
        return mask


@gsregister("reduce")
def select_freqs(
    data: GSData,
    *,
    freq_range: FreqRangeType | None = None,
    indx: np.ndarray | None = None,
    **kwargs,
) -> GSData:
    """Select a subset of the frequency channels."""
    mask = freq_selector(data.freq_array, freq_range=freq_range, indx=indx)

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


def time_selector(
    times: Time,
    loads: list[str],
    *,
    time_range: tuple[Time | float, Time | float] | None = None,
    fmt: str = "jd",
    indx: np.ndarray | None = None,
    load: int | str = "ant",
) -> np.ndarray:
    """Select a subset of the times."""
    if isinstance(load, str):
        load = loads.index(load)

    mask = None
    if time_range is not None:
        if len(time_range) != 2:
            raise ValueError("range must be a length-2 tuple")

        if not isinstance(time_range[0], Time):
            time_range = (
                Time(time_range[0], format=fmt),
                Time(time_range[1], format=fmt),
            )

        t = times[:, load]
        mask = (t >= time_range[0]) & (t <= time_range[1])

    if indx is not None:
        if mask is None:
            mask = np.zeros((len(times),), dtype=bool)
        mask[indx] = True

    return mask


def lst_selector(
    lsts: Longitude,
    loads: list[str],
    *,
    lst_range: LSTRangeType | None = None,
    indx: np.ndarray | None = None,
    load: int | str = "ant",
    gha: bool = False,
) -> GSData:
    """Select a subset of the times."""
    if load == "all":
        load = slice(None)
    if isinstance(load, str):
        load = loads.index(load)

    mask = None
    if lst_range is not None:
        if len(lst_range) != 2:
            raise ValueError("range must be a length-2 tuple")

        if not isinstance(lst_range[0], Longitude):
            lst_range = (
                lst_range[0] % 24 * un.hourangle,
                lst_range[1] % 24 * un.hourangle,
            )

        if gha:
            lsts = lst2gha(lsts)

        # In case we have negative LST/GHA
        lsts = lsts % (24 * un.hourangle)

        if lst_range[0] > lst_range[1]:
            mask = (lsts >= lst_range[1]) & (lsts <= lst_range[0])
        else:
            mask = (lsts >= lst_range[0]) & (lsts <= lst_range[1])

        # Account for the case of load=='all' -- in this case require all loads
        # to be within the range.
        if mask.ndim == 2:
            mask = np.all(mask, axis=1)

    if indx is not None:
        if mask is None:
            mask = np.ones((len(lsts),), dtype=bool)

        if indx.dtype == bool:
            mask[~indx] = False
        else:
            for i in np.arange(len(lsts)):
                if i not in indx:
                    mask[i] = False

    return mask


def load_selector(
    _loads: list[str],
    loads: str | None = None,
    indx: np.ndarray | None = None,
) -> np.ndarray:
    """Select a subset of the loads."""
    mask = np.ones(len(_loads), dtype=bool)

    if indx is not None:
        mask[indx] = True
    if loads is not None:
        mask &= np.isin(_loads, loads)

    return mask


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
        warnings.warn(
            "The 'range' keyword is deprecated, use 'time_range' instead.", stacklevel=2
        )
        time_range = kwargs.pop("range")

    if kwargs:
        raise ValueError(f"Unknown keyword arguments: {kwargs}")

    mask = time_selector(
        data.time_array,
        data.loads,
        time_range=time_range,
        fmt=fmt,
        indx=indx,
        load=load,
    )
    return _mask_times(data, mask)


@gsregister("reduce")
def select_lsts(
    data: GSData,
    *,
    lst_range: LSTRangeType | None = None,
    indx: np.ndarray | None = None,
    load: int | str = "ant",
    gha: bool = False,
    **kwargs,
) -> GSData:
    """Select a subset of the times."""
    if kwargs:
        raise ValueError(f"Unknown keyword arguments: {kwargs}")

    mask = lst_selector(
        data.lst_array, data.loads, lst_range=lst_range, indx=indx, load=load, gha=gha
    )

    return _mask_times(data, mask)


@gsregister("reduce")
def select_loads(
    data: GSData,
    *,
    loads: str | None = None,
    indx: np.ndarray | None = None,
) -> GSData:
    """Select a subset of the loads."""
    mask = load_selector(data.loads, loads, indx)
    return data.update(
        data=data.data[mask],
        loads=[load for i, load in enumerate(data.loads) if mask[i]],
        nsamples=data.nsamples[mask],
        flags={k: v.select(idx=mask, axis="load") for k, v in data.flags.items()},
    )


@gsregister("reduce")
def prune_flagged_integrations(data: GSData, **kwargs) -> GSData:
    """Remove integrations that are flagged for all freq-pol-loads."""
    flg = np.all(data.complete_flags, axis=(0, 1, 3))
    return _mask_times(data, ~flg)
