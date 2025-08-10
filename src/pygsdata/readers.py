"""Module defining the readers for the different file formats."""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Any

import h5py
import hickle
import numpy as np
from astropy import units as un
from astropy.coordinates import Longitude
from astropy.time import Time

from .gsdata import GSData
from .history import History
from .select import (
    freq_selector,
    load_selector,
    lst_selector,
    time_selector,
)
from .telescope import Telescope

GSDATA_READERS = {}


def gsdata_reader(
    formats: list[str],
    select_on_read: bool = False,
):
    """Register a function as a reader for GSData objects."""

    def inner(func):
        func.select_on_read = select_on_read
        func.suffices = formats
        GSDATA_READERS[func.__name__] = func
        return func

    return inner


@gsdata_reader(select_on_read=True, formats=["gsh5"])
def read_gsh5(
    filename: str | Path | h5py.File, selectors: dict[str, Any], group: str = "/"
) -> GSData:
    """Read a GSH5 file to construct the object."""
    if isinstance(filename, str | Path):
        with h5py.File(filename, "r") as fl:
            return read_gsh5(fl[group], selectors)

    version = filename.attrs.get("version", "1.0")

    major = version.split(".")[0]
    reader = getattr(_GSH5Readers, f"v{major}", None)
    if reader is None:
        raise ValueError(f"Unsupported file format version: {version}")

    return reader(filename, selectors)


@gsdata_reader(select_on_read=False, formats=["gspkl"])
def read_gspkl(
    filename: str | Path,
) -> GSData:
    """Read a GSPKL file to construct the object."""
    with Path(filename).open("rb") as fl:
        return pickle.load(fl)


class _GSH5Readers:
    @staticmethod
    def v2(
        fl: h5py.File,
        selectors: dict[str, Any],
    ):
        meta = fl["metadata"]

        telescope = Telescope.from_hdf5(meta["telescope"])
        loads = [x.decode() for x in meta["loads"][()]]

        times = meta["times"][()]
        times = Time(times, format="jd", location=telescope.location)

        time_mask = None
        time_mask = time_selector(
            times,
            loads,
            **selectors.get("time_selector", {}),
        )
        if time_mask is None:
            time_mask = np.ones(len(times), dtype=bool)

        lsts = Longitude(meta["lsts"][()] * un.hourangle)

        lst_mask = lst_selector(
            lsts,
            loads,
            **selectors.get("lst_selector", {}),
        )
        if lst_mask is None:
            lst_mask = np.ones(len(lsts), dtype=bool)

        time_mask &= lst_mask

        extra_kw = {}
        if "time_ranges" in meta:
            time_ranges = meta["time_ranges"][time_mask]
            time_ranges = Time(time_ranges, format="jd", location=telescope.location)
            extra_kw["time_ranges"] = time_ranges

            lst_ranges = Longitude(meta["lst_ranges"][time_mask] * un.hourangle)
            extra_kw["lst_ranges"] = lst_ranges
        else:
            warnings.warn(
                "You wrote this file with a buggy version of pygsdata that did not "
                "include time_ranges and lst_ranges in the file. The time_ranges and "
                "lst_ranges in your object will be set to the default values given "
                "your integration time and times.",
                stacklevel=2,
            )

        freqs = un.Quantity(meta["freqs"][:], unit=meta["freqs"].attrs["unit"])
        freq_mask = freq_selector(freqs, **selectors.get("freq_selector", {}))

        data_unit = meta.attrs["data_unit"]
        objname = meta.attrs["name"]

        intg_time = meta["effective_integration_time"][()] * un.s

        load_mask = load_selector(
            loads,
            **selectors.get("load_selector", {"indx": slice(None)}),
        )

        auxiliary_measurements = {
            name: fl["auxiliary_measurements"][name][time_mask]
            for name in fl["auxiliary_measurements"]
        } or None

        dgrp = fl["data"]
        data = dgrp["data"][load_mask][:, :, time_mask][..., freq_mask]
        nsamples = dgrp["nsamples"][load_mask][:, :, time_mask][..., freq_mask]

        if intg_time.size > 1:
            intg_time = intg_time[load_mask][:, :, time_mask]

        flg_grp = dgrp["flags"]
        flags = {}
        if "names" in flg_grp.attrs:
            flag_keys = flg_grp.attrs["names"]
            for name in flag_keys:
                flags[name] = hickle.load(fl, f"data/flags/{name}")

                if "load" in flags[name].axes and not np.all(load_mask):
                    flags[name] = flags[name].select(idx=load_mask, axis="load")
                if "time" in flags[name].axes and not np.all(time_mask):
                    flags[name] = flags[name].select(idx=time_mask, axis="time")
                if "freq" in flags[name].axes and not np.all(freq_mask):
                    flags[name] = flags[name].select(idx=freq_mask, axis="freq")

        history = History.from_repr(meta.attrs["history"])

        residuals = (
            dgrp["residuals"][load_mask][:, :, time_mask][..., freq_mask]
            if "residuals" in dgrp
            else None
        )

        return GSData(
            data=data,
            times=times[time_mask][:, load_mask],
            lsts=lsts[time_mask][:, load_mask],
            freqs=freqs[freq_mask],
            data_unit=data_unit,
            loads=loads,
            auxiliary_measurements=auxiliary_measurements,
            filename=fl.file.filename,
            nsamples=nsamples,
            flags=flags,
            history=history,
            telescope=telescope,
            residuals=residuals,
            name=objname,
            effective_integration_time=intg_time,
            **extra_kw,
        )
