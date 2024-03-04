"""Module defining the readers for the different file formats."""
from __future__ import annotations

from typing import Any

import h5py
import hickle
import numpy as np
from astropy import units as un
from astropy.coordinates import EarthLocation, Longitude
from astropy.time import Time

from pygsdata.history import History

from ..pygsdata import GSData
from .select import (
    freq_selector,
    load_selector,
    lst_selector,
    time_selector,
)

GSDATA_READERS = {}


def gsdata_reader(
    func: callable,
    formats: list[str],
    select_on_read: bool = False,
):
    """Register a function as a reader for GSData objects."""
    func.select_on_read = select_on_read
    func.suffices = formats
    GSDATA_READERS[func.__name__] = func
    return func


@gsdata_reader(select_on_read=True, formats=["gsh5"])
def read_gsh5(
    filename: str,
    selectors: dict[str, Any],
) -> GSData:
    """Read a GSH5 file to construct the object."""
    with h5py.File(filename, "r") as fl:
        lat, lon, alt = fl["telescope_location"][:]
        telescope_location = EarthLocation(
            lat=lat * un.deg, lon=lon * un.deg, height=alt * un.m
        )
        loads = fl.attrs["loads"].split("|")

        times = fl["time_array"][:]

        if np.all(times < 24.0):
            time_array = Longitude(times * un.hour)
        else:
            time_array = Time(times, format="jd", location=telescope_location)

        time_mask = None
        if isinstance(time_array, Time):
            time_mask = time_selector(
                time_array,
                loads,
                **selectors.get("time_selector", {}),
            )
        if time_mask is None:
            time_mask = np.ones(len(time_array), dtype=bool)

        if isinstance(time_array, Time):
            lsts = time_array.sidereal_time("apparent")
        else:
            lsts = time_array

        lst_mask = lst_selector(
            lsts,
            loads,
            **selectors.get("lst_selector", {}),
        )
        if lst_mask is None:
            lst_mask = np.ones(len(lsts), dtype=bool)

        time_mask &= lst_mask

        freq_array = fl["freq_array"][:] * un.MHz
        freq_mask = freq_selector(
            freqs=freq_array, **selectors.get("freq_selector", {})
        )

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
            time_array=time_array,
            freq_array=freq_array,
            data_unit=data_unit,
            loads=loads,
            auxiliary_measurements=auxiliary_measurements,
            filename=filename,
            nsamples=nsamples,
            flags=flags,
            history=history,
            telescope_location=telescope_location,
            residuals=residuals,
            name=objname,
        )
