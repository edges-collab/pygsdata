"""Concatenation methods for GSData objects."""

from __future__ import annotations

import functools
import operator
import warnings
from collections.abc import Sequence

import numpy as np

from .gsdata import GSData
from .register import gsregister


def concat_times(data: Sequence[GSData]) -> GSData:
    """Concatenate GSData objects along the time axis."""
    times = np.concatenate([d.time_array for d in data], axis=0)
    data_array = np.concatenate([d.data for d in data], axis=2)
    nsamples = np.concatenate([d.nsamples for d in data], axis=2)
    time_ranges = np.concatenate([d.time_ranges for d in data], axis=0)

    for d in data:
        if set(d.flags.keys()) != set(data[0].flags.keys()):
            raise ValueError(
                "Flags must have the same keys to do concatenation, got "
                f"{d.flags.keys()} and {data[0].flags.keys()}"
            )

    flags = {}
    for k, flg in data[0].flags.items():
        if "time" in flg.axes:
            flags[k] = flg.concat([d.flags for d in data[1:]], axis="time")
        else:
            warnings.warn(
                f"Flags {k} do not have a time axis, so they will not be concatenated.",
                stacklevel=2,
            )
            flags[k] = flg

    if data[0].residuals is not None:
        residuals = np.concatenate([d.residuals for d in data], axis=2)
    else:
        residuals = None

    return data[0].update(
        data=data_array,
        time_array=times,
        time_ranges=time_ranges,
        nsamples=nsamples,
        flags=flags,
        residuals=residuals,
    )


def concat_freqs(data: Sequence[GSData]) -> GSData:
    """Concatenate GSData objects along the time axis."""
    freqs = np.concatenate([d.freq_array for d in data])
    data_array = np.concatenate([d.data for d in data], axis=-1)
    nsamples = np.concatenate([d.nsamples for d in data], axis=-1)

    for d in data:
        if set(d.flags.keys()) != set(data[0].flags.keys()):
            raise ValueError(
                "Flags must have the same keys to do concatenation, got "
                f"{d.flags.keys()} and {data[0].flags.keys()}"
            )

    flags = {}
    for k, flg in data[0].flags.items():
        if "freq" in flg.axes:
            flags[k] = flg.concat([d.flags for d in data[1:]], axis="freq")
        else:
            warnings.warn(
                f"Flags {k} do not have a freq axis, so they will not be concatenated.",
                stacklevel=2,
            )
            flags[k] = flg

    if data[0].residuals is not None:
        residuals = np.concatenate([d.residuals for d in data], axis=-1)
    else:
        residuals = None

    return data[0].update(
        data=data_array,
        freq_array=freqs,
        nsamples=nsamples,
        flags=flags,
        residuals=residuals,
    )


def concat_loads(data: Sequence[GSData]) -> GSData:
    """Concatenate GSData objects along the time axis."""
    loads = functools.reduce(operator.iadd, (list(d.loads) for d in data), [])
    data_array = np.concatenate([d.data for d in data], axis=0)
    nsamples = np.concatenate([d.nsamples for d in data], axis=0)

    for d in data:
        if set(d.flags.keys()) != set(data[0].flags.keys()):
            raise ValueError(
                "Flags must have the same keys to do concatenation, got "
                f"{d.flags.keys()} and {data[0].flags.keys()}"
            )

    flags = {}
    for k, flg in data[0].flags.items():
        if "load" in flg.axes:
            flags[k] = flg.concat([d.flags for d in data[1:]], axis="load")
        else:
            warnings.warn(
                f"Flags {k} do not have a load axis, so they will not be concatenated.",
                stacklevel=2,
            )
            flags[k] = flg

    if data[0].residuals is not None:
        residuals = np.concatenate([d.residuals for d in data], axis=0)
    else:
        residuals = None

    return data[0].update(
        data=data_array,
        loads=loads,
        nsamples=nsamples,
        flags=flags,
        residuals=residuals,
    )


@gsregister("gather")
def concat(
    data: Sequence[GSData],
    axis: str = "time",
) -> GSData:
    """Concatenate GSData objects along the time axis."""
    if axis == "time":
        return concat_times(data)
    elif axis == "freq":
        return concat_freqs(data)
    elif axis == "load":
        return concat_loads(data)
    else:
        raise ValueError(f"Axis must be 'time', 'freq' or 'load', got {axis}")
