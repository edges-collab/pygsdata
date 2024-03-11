"""Module defining the readers for the different file formats."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import hickle
import numpy as np
from astropy import units as un
from astropy.coordinates import EarthLocation, Longitude
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
    filename: str | Path | h5py.File,
    selectors: dict[str, Any],
) -> GSData:
    """Read a GSH5 file to construct the object."""
    if isinstance(filename, (str, Path)):
        with h5py.File(filename, "r") as fl:
            return read_gsh5(fl, selectors)

    version = fl.attrs["version"]
    major = version.split(".")[0]
    reader = getattr(_GSH5Readers, f"v{major}", None)
    if reader is None:
        raise ValueError(f"Unsupported file format version: {version}")

    return reader(fl, selectors)


class _GSH5Readers:
    @staticmethod
    def v1(
        fl: h5py.File,
        selectors: dict[str, Any],
    ) -> GSData:
        """Read a GSH5 file to construct the object."""
        lat, lon, alt = fl["telescope_location"][:]
        telescope_location = EarthLocation(
            lat=lat * un.deg, lon=lon * un.deg, height=alt * un.m
        )
        loads = fl.attrs["loads"].split("|")

        times = fl["times"][:]

        if np.all(times < 24.0):
            times = Longitude(times * un.hour)
        else:
            times = Time(times, format="jd", location=telescope_location)

        time_mask = None
        if isinstance(times, Time):
            time_mask = time_selector(
                times,
                loads,
                **selectors.get("time_selector", {}),
            )
        if time_mask is None:
            time_mask = np.ones(len(times), dtype=bool)

        lsts = times.sidereal_time("apparent") if isinstance(times, Time) else times

        lst_mask = lst_selector(
            lsts,
            loads,
            **selectors.get("lst_selector", {}),
        )
        if lst_mask is None:
            lst_mask = np.ones(len(lsts), dtype=bool)

        time_mask &= lst_mask

        freqs = fl["freqs"][:] * un.MHz
        freq_mask = freq_selector(freqs=freqs, **selectors.get("freq_selector", {}))

        data_unit = fl.attrs["data_unit"]
        objname = fl.attrs["name"]

        load_mask = load_selector(
            loads=loads,
            **selectors.get("load_selector", {}),
        )

        auxiliary_measurements = {
            name: fl["auxiliary_measurements"][name][:]
            for name in fl["auxiliary_measurements"]
        }

        data = fl["data"][load_mask, :, time_mask, freq_mask]
        nsamples = fl["nsamples"][load_mask, :, time_mask, freq_mask]

        flg_grp = fl["flags"]
        flags = {}
        if "names" in flg_grp.attrs:
            flag_keys = flg_grp.attrs["names"]
            for name in flag_keys:
                flags[name] = hickle.load(fl["flags"][name])

                if "load" in flags.axes and not np.all(load_mask):
                    flags[name] = flags[name].select(idx=load_mask, axis="load")
                if "time" in flags.axes and not np.all(time_mask):
                    flags[name] = flags[name].select(idx=time_mask, axis="time")
                if "freq" in flags.axes and not np.all(freq_mask):
                    flags[name] = flags[name].select(idx=freq_mask, axis="freq")

        # TODO: Have to select flags somehow
        history = History.from_repr(fl.attrs["history"])

        residuals = (
            fl["residuals"][load_mask, :, time_mask, freq_mask]
            if "residuals" in fl
            else None
        )

        return GSData(
            data=data,
            times=times,
            freqs=freqs,
            data_unit=data_unit,
            loads=loads,
            auxiliary_measurements=auxiliary_measurements,
            filename=fl.filename,
            nsamples=nsamples,
            flags=flags,
            history=history,
            telescope_location=telescope_location,
            residuals=residuals,
            name=objname,
        )

    @staticmethod
    def v2(
        fl: h5py.File,
        selectors: dict[str, Any],
    ):
        telescope = Telescope.from_hdf5(fl["telescope"])

        meta = fl["metadata"]
        loads = meta["loads"][()]
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

        lsts = meta["lsts"][()]

        lst_mask = lst_selector(
            lsts,
            loads,
            **selectors.get("lst_selector", {}),
        )
        if lst_mask is None:
            lst_mask = np.ones(len(lsts), dtype=bool)

        time_mask &= lst_mask

        freqs = un.Quantity(meta["freqs"][:], unit=meta["freqs"].attrs["unit"])
        freq_mask = freq_selector(freqs=freqs, **selectors.get("freq_selector", {}))

        data_unit = meta.attrs["data_unit"]
        objname = meta.attrs["name"]

        load_mask = load_selector(
            loads=loads,
            **selectors.get("load_selector", {}),
        )

        auxiliary_measurements = {
            name: fl["auxiliary_measurements"][name][:]
            for name in fl["auxiliary_measurements"]
        }

        dgrp = fl["data"]
        data = dgrp["data"][load_mask, :, time_mask, freq_mask]
        nsamples = dgrp["nsamples"][load_mask, :, time_mask, freq_mask]

        flg_grp = dgrp["flags"]
        flags = {}
        if "names" in flg_grp.attrs:
            flag_keys = flg_grp.attrs["names"]
            for name in flag_keys:
                flags[name] = hickle.load(fl["flags"][name])

                if "load" in flags.axes and not np.all(load_mask):
                    flags[name] = flags[name].select(idx=load_mask, axis="load")
                if "time" in flags.axes and not np.all(time_mask):
                    flags[name] = flags[name].select(idx=time_mask, axis="time")
                if "freq" in flags.axes and not np.all(freq_mask):
                    flags[name] = flags[name].select(idx=freq_mask, axis="freq")

        history = History.from_repr(meta.attrs["history"])

        residuals = (
            dgrp["residuals"][load_mask, :, time_mask, freq_mask]
            if "residuals" in dgrp
            else None
        )

        return GSData(
            data=data,
            times=times,
            lsts=lsts,
            freqs=freqs,
            data_unit=data_unit,
            loads=loads,
            auxiliary_measurements=auxiliary_measurements,
            filename=fl.filename,
            nsamples=nsamples,
            flags=flags,
            history=history,
            telescope=telescope,
            residuals=residuals,
            name=objname,
        )
