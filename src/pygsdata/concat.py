"""Concatenation methods for GSData objects."""

from __future__ import annotations

import functools
import operator
import warnings
from collections.abc import Sequence

import numpy as np
from astropy.table import vstack

from .gsdata import GSData
from .register import gsregister
from .utils import time_concat


@gsregister("gather")
def concat(
    data: Sequence[GSData],
    axis: str = "time",
) -> GSData:
    """Concatenate GSData objects along the time axis."""
    try:
        axnum = {"time": 2, "freq": 3, "load": 0, "pol": 1}[axis]
    except KeyError as e:
        raise ValueError(
            f"Axis must be 'time', 'freq', 'load' or 'pol', got {axis}"
        ) from e

    data_array = np.concatenate([d.data for d in data], axis=axnum)
    nsamples = np.concatenate([d.nsamples for d in data], axis=axnum)

    for d in data:
        if set(d.flags.keys()) != set(data[0].flags.keys()):
            raise ValueError(
                "Flags must have the same keys to do concatenation, got "
                f"{d.flags.keys()} and {data[0].flags.keys()}"
            )

    flags = {}
    for k, flg in data[0].flags.items():
        if axis in flg.axes:
            flags[k] = flg.concat([d.flags[k] for d in data[1:]], axis=axis)
        else:
            warnings.warn(
                f"Flags {k} do not have a {axis} axis. They will not be concatenated.",
                stacklevel=2,
            )
            flags[k] = flg

    if data[0].residuals is not None:
        residuals = np.concatenate([d.residuals for d in data], axis=axnum)
    else:
        residuals = None

    kw = {}
    if axis == "time":
        kw["times"] = time_concat([d.times for d in data], axis=0)
        kw["lsts"] = np.concatenate([d.lsts for d in data], axis=0)
        kw["time_ranges"] = time_concat([d.time_ranges for d in data], axis=0)
        kw["lst_ranges"] = np.concatenate([d.lst_ranges for d in data], axis=0)
        kw["auxiliary_measurements"] = vstack([d.auxiliary_measurements for d in data])
    elif axis == "freq":
        kw["freqs"] = np.concatenate([d.freqs for d in data])
    elif axis == "load":
        kw["loads"] = functools.reduce(operator.iadd, (list(d.loads) for d in data), [])
        kw["times"] = time_concat([d.times for d in data], axis=1)
        kw["time_ranges"] = time_concat([d.time_ranges for d in data], axis=1)
        kw["lsts"] = np.concatenate([d.lsts for d in data], axis=1)
        kw["lst_ranges"] = np.concatenate([d.lst_ranges for d in data], axis=1)
    elif axis == "pol":
        kw["pols"] = functools.reduce(operator.iadd, (list(d.pols) for d in data), [])

    return data[0].update(
        data=data_array,
        nsamples=nsamples,
        flags=flags,
        residuals=residuals,
        **kw,
    )
