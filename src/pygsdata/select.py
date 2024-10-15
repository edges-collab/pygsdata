"""Selection functions for GSData objects."""

from __future__ import annotations

import warnings

import numpy as np
from astropy import units as un
from astropy.coordinates import Angle, Longitude
from astropy.time import Time

from .coordinates import lst2gha
from .gsdata import GSData
from .register import gsregister
from .types import FreqRangeType, FreqType, LSTRangeType


def freq_selector(
    _freqs: FreqType,
    *,
    freq_range: FreqRangeType | None = None,
    indx: np.ndarray | slice | None = None,
) -> np.ndarray:
    """Select a subset of the frequency channels."""
    mask = None
    if freq_range is not None:
        if not isinstance(freq_range[0], un.Quantity):
            warnings.warn(
                "frequency range given without units, assuming MHz.", stacklevel=2
            )
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
) -> GSData:
    """Select a subset of the frequency channels in a GSData object.

    Parameters
    ----------
    data
        The GSData object to downselect.
    freq_range
        A two-tuple specifying the frequency range to select. The elements should
        each be astropy quantities with frequency-type units. For example
        `(50*un.MHz, 100*un.MHz)`. If the values are simply floats (e.g. `(50, 100)`),
        a warning will be raised, and they will be assumed to be in MHz.
        This parameter is **optional** -- you can instead specify ``indx`` if your
        selection is more involved.
    indx
        An object that can slice a numpy array (i.e. a list of integers, a numpy array
        of integers, or a slice object). This will be used to directly select the
        frequency channels. This parameter is **optional** -- you can instead
        specify ``freq_range``. If *both* are specified, the returned frequencies will
        be the *union* of the selectors (i.e. in the freq range OR in the indx).

    Notes
    -----
    If both ``freq_range`` and ``indx`` are specified, they will be OR-ed, i.e. the
    returned frequencies will be *either* in the freq range or in the indx. To achieve
    and-like behavior, simply use the selector twice.
    """
    mask = freq_selector(data.freqs, freq_range=freq_range, indx=indx)

    return data.update(
        data=data.data[..., mask],
        freqs=data.freqs[mask],
        nsamples=data.nsamples[..., mask],
        flags={k: v.select(idx=mask, axis="freq") for k, v in data.flags.items()},
    )


def _mask_times(data: GSData, mask: np.ndarray) -> GSData:
    if mask is None:
        return data

    return data.update(
        data=data.data[:, :, mask],
        times=data.times[mask],
        time_ranges=data.time_ranges[mask],
        auxiliary_measurements=data.auxiliary_measurements[mask]
        if data.auxiliary_measurements is not None
        else None,
        nsamples=data.nsamples[:, :, mask],
        flags={k: v.select(idx=mask, axis="time") for k, v in data.flags.items()},
        residuals=data.residuals[:, :, mask] if data.residuals is not None else None,
        effective_integration_time=data.effective_integration_time[:, :, mask],
        lsts=data.lsts[mask],
        lst_ranges=data.lst_ranges[mask],
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

    lsts = lsts[:, load]

    mask = None
    if lst_range is not None:
        if len(lst_range) != 2:
            raise ValueError("range must be a length-2 tuple")

        if not isinstance(lst_range[0], (Longitude, Angle)):
            lst_range = (
                lst_range[0] % 24 * un.hourangle,
                lst_range[1] % 24 * un.hourangle,
            )
            if lst_range[1] == 0 * un.hourangle and lst_range[1] <= lst_range[0]:
                lst_range = (lst_range[0], 24 * un.hourangle)
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
            mask = np.zeros((len(lsts),), dtype=bool)

        mask[indx] = True

    return mask


def load_selector(
    _loads: list[str],
    loads: str | None = None,
    indx: np.ndarray | list | None = None,
) -> np.ndarray:
    """Select a subset of the loads."""
    mask = np.zeros(len(_loads), dtype=bool)

    if indx is not None:
        mask[indx] = True

    if loads is not None:
        mask |= np.isin(_loads, loads)

    return mask


@gsregister("reduce")
def select_times(
    data: GSData,
    *,
    time_range: tuple[Time | float, Time | float] | None = None,
    fmt: str = "jd",
    indx: np.ndarray | None = None,
    load: int | str = "ant",
) -> GSData:
    """Select a subset of the times."""
    mask = time_selector(
        data.times,
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
        data.lsts, data.loads, lst_range=lst_range, indx=indx, load=load, gha=gha
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
        times=data.times[:, mask],
        time_ranges=data.time_ranges[:, mask],
        lsts=data.lsts[:, mask],
        lst_ranges=data.lst_ranges[:, mask],
        effective_integration_time=data.effective_integration_time[mask],
    )


@gsregister("reduce")
def prune_flagged_integrations(data: GSData, **kwargs) -> GSData:
    """Remove integrations that are flagged for all freq-pol-loads."""
    flg = np.all(data.complete_flags, axis=(0, 1, 3))
    return _mask_times(data, ~flg)
