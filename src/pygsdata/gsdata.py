"""
A module containing the class GSData, a variant of UVData specific to single antennas.

The GSData object simplifies handling of radio astronomy data taken from a single
antenna, adding self-consistent metadata along with the data itself, and providing
key methods for data selection, I/O, and analysis.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable
from functools import cached_property
from pathlib import Path
from typing import Any, Literal

import astropy.units as un
import h5py
import hickle
import numpy as np
from astropy.coordinates import Longitude
from astropy.table import QTable
from astropy.time import Time
from attrs import cmp_using, define, evolve, field
from attrs import converters as cnv
from attrs import validators as vld

from . import coordinates as crd
from .attrs import cmp_qtable, lstfield, npfield, timefield
from .gsflag import GSFlag
from .history import History, Stamp
from .telescope import Telescope, _pol_converter
from .utils import time_concat

logger = logging.getLogger(__name__)


@define(slots=False)
class GSData:
    """A generic container for Global-Signal data.

    Parameters
    ----------
    data
        The data array (i.e. what the telescope measures). This must be a 4D array whose
        dimensions are (load, polarization, time, frequency). The data can be raw
        powers, calibrated temperatures, or even model residuals to such. Their type is
        specified by the ``data_unit`` attribute.
    freqs
        The frequency array. This must be a 1D array of frequencies specified as an
        astropy Quantity.
    times
        The time array. This must be a 2D array of shape (times, loads). It can be in
        one of two formats: either an astropy Time object, specifying the absolute time,
        or an astropy Longitude object, specying the LSTs. In "lst" mode, there are
        many methods that become unavailable.
    telescope_location
        The telescope location. This must be an astropy EarthLocation object.
    loads
        The names of the loads. Usually there is a single load ("ant"), but arbitrary
        loads may be specified.
    nsamples
        An array with the same shape as the data array, specifying the number of samples
        that go into each data point. This is unitless, and can be used with the
        ``effective_integration_time`` attribute to compute the total effective
        integration time going into any measurement.
    effective_integration_time
        An astropy Quantity that specifies the amount of time going into a single
        "sample" of the data. This can either be a scalar, or a 4D array with the same
        shape as the data array. If it is a scalar, it is assumed to be the same for all
        data points. The default value is the integration time of the telescope.
        Note that this value is *only* meant to be used to track the expected noise
        level in the data, in conjunction with nsamples. It is not checked for
        whether the time_ranges match the integration time (since the effective time
        can be smaller than the time range due to windowing, or because the time_range
        includes multiple observations).
    flags
        A dictionary mapping filter names to boolean arrays. Each boolean array has the
        same shape as the data array, and is True where the data is flagged.
    history
        A tuple of dictionaries, each of which is a record of a previous processing
        step.
    telescope_name
        The name of the telescope.
    residuals
        An optional array of the same shape as data that holds the residuals of a model
        fit to the data.
    auxiliary_measurements
        A dictionary mapping measurement names to arrays. Each array must have its
        leading axis be the same length as the time array.
    filename
        The filename from which the data was read (if any). Used for writing additional
        data if more is added (eg. flags, data model).
    """

    telescope: Telescope = field(validator=vld.instance_of(Telescope))
    data: np.ndarray = npfield(dtype=float, possible_ndims=(4,))
    freqs: un.Quantity[un.MHz] = npfield(possible_ndims=(1,), unit=un.MHz)
    times: Time = timefield(possible_ndims=(2,))

    pols: tuple[str] = field(converter=_pol_converter)
    _effective_integration_time: un.Quantity[un.s] = npfield(
        possible_ndims=(0, 3), unit=un.s
    )

    nsamples: np.ndarray = npfield(dtype=float, possible_ndims=(4,))
    loads: tuple[str] = field(converter=tuple)

    flags: dict[str, GSFlag] = field(factory=dict)

    history: History = field(
        factory=History, validator=vld.instance_of(History), eq=False
    )
    residuals: np.ndarray | None = npfield(
        default=None, possible_ndims=(4,), dtype=float
    )

    data_unit: Literal["power", "temperature", "uncalibrated", "uncalibrated_temp"] = (
        field(default="power")
    )
    auxiliary_measurements: QTable | None = field(
        default=None, converter=cnv.optional(QTable), eq=cmp_using(cmp_qtable)
    )
    time_ranges: Time = timefield(shape=(None, None, 2))
    lsts: Longitude = lstfield(possible_ndims=(2,))
    lst_ranges: Longitude = lstfield(possible_ndims=(3,))

    filename: Path | None = field(default=None, converter=cnv.optional(Path), eq=False)
    _file_appendable: bool = field(default=True, converter=bool)
    name: str = field(default="", converter=str)

    @nsamples.validator
    def _nsamples_validator(self, attribute, value):
        if value.shape != self.data.shape:
            raise ValueError("nsamples must have the same shape as data")

    @nsamples.default
    def _nsamples_default(self) -> np.ndarray:
        return np.ones_like(self.data)

    @flags.validator
    def _flags_validator(self, attribute, value):
        if not isinstance(value, dict):
            raise TypeError("flags must be a dict")

        for key, flag in value.items():
            if not isinstance(flag, GSFlag):
                raise TypeError("flags values must be GSFlag instances")

            flag._check_compat(self)

            if not isinstance(key, str):
                raise ValueError("flags keys must be strings")

    @residuals.validator
    def _residuals_validator(self, attribute, value):
        if value is not None and value.shape != self.data.shape:
            raise ValueError("residuals must have the same shape as data")

    @freqs.validator
    def _freqs_validator(self, attribute, value):
        if value.shape != (self.nfreqs,):
            raise ValueError(
                "freqs must have the size nfreqs. "
                f"Got {value.shape} instead of {self.nfreqs}"
            )

    @times.validator
    def _times_validator(self, attribute, value):
        if value.shape != (self.ntimes, self.nloads):
            raise ValueError(
                f"times must have the size (ntimes, nloads), got {value.shape} "
                f"instead of {(self.ntimes, self.nloads)}"
            )

    @pols.default
    def _pols_default(self) -> tuple[str]:
        return self.telescope.pols

    @time_ranges.default
    def _time_ranges_default(self):
        return time_concat(
            (
                self.times[:, :, None],
                self.times[:, :, None] + self.telescope.integration_time,
            ),
            axis=-1,
        )

    @time_ranges.validator
    def _time_ranges_validator(self, attribute, value):
        if value.shape != (self.ntimes, self.nloads, 2):
            raise ValueError(
                f"time_ranges must have the size (ntimes, nloads, 2), got {value.shape}"
                f" instead of {(self.ntimes, self.nloads, 2)}."
            )

        if not np.all((value[..., 1] - value[..., 0]).value > 0):
            # TODO: properly check lst-type input, which can wrap...
            raise ValueError("time_ranges must all be greater than zero")

    @loads.default
    def _loads_default(self) -> tuple[str]:
        if self.nloads == 1:
            return ("ant",)
        elif self.nloads == 3:
            return ("ant", "internal_load", "internal_load_plus_noise_source")
        else:
            raise ValueError(
                "If data has more than one source, loads must be specified"
            )

    @loads.validator
    def _loads_validator(self, attribute, value):
        if len(value) != self.data.shape[0]:
            raise ValueError(
                "loads must have the same length as the number of loads in data"
            )

        if not all(isinstance(x, str) for x in value):
            raise ValueError("loads must be a tuple of strings")

    @auxiliary_measurements.validator
    def _aux_meas_vld(self, attribute, value):
        if value is None:
            return

        if len(value) != self.ntimes:
            raise ValueError(
                "auxiliary_measurements must be length ntimes."
                f" Got {len(value)} instead of {self.ntimes}."
            )

    @_effective_integration_time.default
    def _eff_int_time_default(self) -> un.Quantity[un.s]:
        return self.telescope.integration_time * np.ones(
            (self.nloads, self.npols, self.ntimes)
        )

    @_effective_integration_time.validator
    def _eff_int_time_vld(self, attribute, value):
        if np.any(value.value <= 0):
            raise ValueError("effective_integration_time must be greater than zero")

        if value.size != 1 and value.shape != (self.nloads, self.npols, self.ntimes):
            raise ValueError(
                "effective_integration_time must be a scalar or have shape "
                f"(nloads, npols, ntimes), got {value.shape}"
            )

    @cached_property
    def effective_integration_time(self) -> un.Quantity[un.s]:
        """The effective integration time."""
        if self._effective_integration_time.size == 1:
            return self._effective_integration_time * np.ones(self.data.shape[:-1])

        return self._effective_integration_time

    @data_unit.validator
    def _data_unit_validator(self, attribute, value):
        if value not in (
            "power",
            "temperature",
            "uncalibrated",
            "uncalibrated_temp",
        ):
            raise ValueError(
                'data_unit must be one of "power", "temperature", "uncalibrated",'
                '"uncalibrated_temp"'
            )

    @property
    def nfreqs(self) -> int:
        """The number of frequency channels."""
        return self.data.shape[-1]

    @property
    def nloads(self) -> int:
        """The number of loads."""
        return self.data.shape[0]

    @property
    def ntimes(self) -> int:
        """The number of times."""
        return self.data.shape[-2]

    @property
    def npols(self) -> int:
        """The number of polarizations."""
        return self.data.shape[1]

    @property
    def model(self) -> np.ndarray | None:
        """The model of the data."""
        if self.residuals is None:
            return None

        return self.data - self.residuals

    @lsts.default
    def _lsts_default(self) -> Longitude:
        return self.times.sidereal_time("apparent", self.telescope.location)

    @lsts.validator
    def _lsts_validator(self, attribute, value):
        if value.shape != (self.ntimes, self.nloads):
            raise ValueError(
                f"lsts must have the size (ntimes, nloads), got {value.shape} "
                f"instead of {(self.ntimes, self.nloads)}"
            )

    @lst_ranges.default
    def _lst_ranges_default(self) -> Longitude:
        return self.time_ranges.sidereal_time("apparent", self.telescope.location)

    @lst_ranges.validator
    def _lst_ranges_validator(self, attribute, value):
        if value.shape != (self.ntimes, self.nloads, 2):
            raise ValueError(
                f"lst_ranges must have the size (ntimes, nloads, 2), got {value.shape} "
                f"instead of {(self.ntimes, self.nloads, 2)}"
            )

    @classmethod
    def from_file(
        cls,
        filename: str | Path,
        reader: str | None = None,
        selectors: dict[str, Any] | None = None,
        concat_axis: Literal["load", "pol", "time", "freq"] | None = None,
        **kw,
    ) -> GSData:
        """Create a GSData instance from a file.

        This method attempts to auto-detect the file type and read it.
        """
        from .readers import GSDATA_READERS

        selectors = selectors or {}

        def _from_file(pth, reader):
            filename = Path(pth)

            if reader is None:
                reader = filename.suffix[1:]

            fnc = next(
                (k for k in GSDATA_READERS.values() if reader in k.suffices), None
            )

            if fnc is None:
                raise ValueError(f"Unrecognized file type {reader}")

            if fnc.select_on_read:
                return fnc(filename, selectors=selectors, **kw)

            from .select import select_freqs, select_loads, select_lsts, select_times

            data = fnc(filename, **kw)

            if "freq_selector" in selectors:
                data = select_freqs(data, **selectors["freq_selector"])
            if "time_selector" in selectors:
                data = select_times(data, **selectors["time_selector"])
            if "lst_selector" in selectors:
                data = select_lsts(data, **selectors["lst_selector"])
            if "load_selector" in selectors:
                data = select_loads(data, **selectors["load_selector"])

            return data

        filename = [filename] if isinstance(filename, (str, Path)) else filename
        datas = [_from_file(pth, reader) for pth in filename]

        if len(datas) == 1:
            return datas[0]

        from .concat import concat

        return concat(datas, concat_axis)

    def write_gsh5(self, filename: str) -> GSData:
        """Write the data in the GSData object to a GSH5 file."""
        with h5py.File(filename, "w") as fl:
            # The GSH5 file version: <major>.<minor>. The minor version is incremented
            # when the file format changes in a backwards-compatible way. The major
            # version is incremented when the file format changes in a way
            # that requires a new reader.
            fl.attrs["version"] = "2.1"

            meta = fl.create_group("metadata")
            self.telescope.write(meta.create_group("telescope"))
            meta["freqs"] = self.freqs.to_value("MHz")
            meta["freqs"].attrs["unit"] = "MHz"
            meta["effective_integration_time"] = (
                self._effective_integration_time.to_value("s")
            )

            meta["times"] = self.times.jd
            meta["time_ranges"] = self.time_ranges.jd
            meta["lsts"] = self.lsts.hour
            meta["lst_ranges"] = self.lst_ranges.hour
            meta.attrs["data_unit"] = self.data_unit
            meta["loads"] = self.loads
            meta.attrs["history"] = repr(self.history)
            meta.attrs["name"] = self.name

            dgrp = fl.create_group("data")
            dgrp["data"] = self.data
            dgrp["nsamples"] = self.nsamples

            flg_grp = dgrp.create_group("flags")
            if self.flags:
                flg_grp.attrs["names"] = tuple(self.flags.keys())
                for name, flag in self.flags.items():
                    hickle.dump(flag, flg_grp.create_group(name))

            # Data model
            if self.residuals is not None:
                dgrp["residuals"] = self.residuals

            # Now aux measurements
            aux_grp = fl.create_group("auxiliary_measurements")
            if self.auxiliary_measurements is not None:
                for name, meas in self.auxiliary_measurements.items():
                    aux_grp[name] = meas

        return self.update(filename=filename)

    def update(self, **kwargs):
        """Return a new GSData object with updated attributes."""
        # If the user passes a single dictionary as history, append it.
        # Otherwise raise an error, unless it's not passed at all.
        history = kwargs.pop("history", None)
        if isinstance(history, Stamp):
            history = self.history.add(history)
        elif isinstance(history, dict):
            history = self.history.add(Stamp(**history))
        elif history is not None:
            raise ValueError("History must be a Stamp object or dictionary")
        else:
            history = self.history

        return evolve(self, history=history, **kwargs)

    def __add__(self, other: GSData) -> GSData:
        """Add two GSData objects."""
        if not isinstance(other, GSData):
            raise TypeError("can only add GSData objects")

        if self.data.shape != other.data.shape:
            raise ValueError("Cannot add GSData objects with different shapes")

        if not np.allclose(self.freqs, other.freqs):
            raise ValueError("Cannot add GSData objects with different frequencies")

        if self.auxiliary_measurements and not other.auxiliary_measurements:
            aux = self.auxiliary_measurements
        elif not self.auxiliary_measurements and other.auxiliary_measurements:
            aux = other.auxiliary_measurements
        elif self.auxiliary_measurements:
            aux = dict(other.auxiliary_measurements.items())
            aux.update(self.auxiliary_measurements)
            aux = QTable(aux)
            if any(
                k in other.auxiliary_measurements for k in self.auxiliary_measurements
            ):
                warnings.warn(
                    "Overlapping auxiliary measurements exist between objects,"
                    " the ones in the first object will be retained.",
                    stacklevel=2,
                )
        else:
            aux = None

        if not np.allclose(self.times.jd, other.times.jd, rtol=0, atol=1e-8):
            raise ValueError("Cannot add GSData objects with different times")

        # If non of the above, then we have two GSData objects at the same times and
        # frequencies. Adding them should just be a weighted sum.
        nsamples = self.flagged_nsamples + other.flagged_nsamples
        d1 = np.ma.masked_array(self.data, mask=self.complete_flags)
        d2 = np.ma.masked_array(other.data, mask=other.complete_flags)

        mean = self.flagged_nsamples * d1 + other.flagged_nsamples * d2

        if self.residuals is not None and other.residuals is not None:
            r1 = np.ma.masked_array(self.residuals, mask=self.complete_flags)
            r2 = np.ma.masked_array(other.residuals, mask=other.complete_flags)
            resids = (
                self.flagged_nsamples * r1 + other.flagged_nsamples * r2
            ) / nsamples
        else:
            resids = None

        total_flags = GSFlag(flags=self.complete_flags & other.complete_flags)
        return self.update(
            data=mean.data,
            residuals=resids,
            nsamples=nsamples,
            flags={"summed_flags": total_flags},
            auxiliary_measurements=aux,
        )

    @cached_property
    def gha(self) -> np.ndarray:
        """The GHA's of the observations."""
        return crd.lst2gha(self.lsts)

    def get_moon_azel(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the Moon's azimuth and elevation for each time in deg."""
        return crd.moon_azel(
            self.times[:, self.loads.index("ant")], self.telescope.location
        )

    def get_sun_azel(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the Sun's azimuth and elevation for each time in deg."""
        return crd.sun_azel(
            self.times[:, self.loads.index("ant")], self.telescope.location
        )

    @property
    def nflagging_ops(self) -> int:
        """Returns the number of flagging operations."""
        return len(self.flags)

    def get_cumulative_flags(
        self, which_flags: tuple[str] | None = None, ignore_flags: tuple[str] = ()
    ) -> np.ndarray:
        """Return accumulated flags."""
        if which_flags is None:
            which_flags = self.flags.keys()
        elif not which_flags or not self.flags:
            return np.zeros(self.data.shape, dtype=bool)

        which_flags = tuple(s for s in which_flags if s not in ignore_flags)
        if not which_flags:
            return np.zeros(self.data.shape, dtype=bool)

        flg = self.flags[which_flags[0]].full_rank_flags
        for k in which_flags[1:]:
            flg = flg | self.flags[k].full_rank_flags

        # Get into full data-shape
        if flg.shape != self.data.shape:
            flg = flg | np.zeros(self.data.shape, dtype=bool)

        return flg

    @cached_property
    def complete_flags(self) -> np.ndarray:
        """Returns the complete flag array."""
        return self.get_cumulative_flags()

    def get_flagged_nsamples(
        self, which_flags: tuple[str] | None = None, ignore_flags: tuple[str] = ()
    ) -> np.ndarray:
        """Get the nsamples of the data after accounting for flags."""
        cumflags = self.get_cumulative_flags(which_flags, ignore_flags)
        return self.nsamples * (~cumflags).astype(int)

    @cached_property
    def flagged_nsamples(self) -> np.ndarray:
        """Weights accounting for all flags."""
        return self.get_flagged_nsamples()

    def get_initial_yearday(self, hours: bool = False, minutes: bool = False) -> str:
        """Return the year-day representation of the first time-sample in the data."""
        if minutes and not hours:
            raise ValueError("Cannot return minutes without hours")

        subfmt = "date_hm" if hours else "date"

        out = self.times[0, self.loads.index("ant")].to_value("yday", subfmt)

        if hours and not minutes:
            out = ":".join(out.split(":")[:-1])

        return out

    def add_flags(
        self,
        filt: str,
        flags: np.ndarray | GSFlag | Path,
        append_to_file: bool | None = None,
    ):
        """Append a set of flags to the object and optionally append them to file.

        You can always write out a *new* file, but appending flags is non-destructive,
        and so we allow it to be appended, in order to save disk space and I/O.
        """
        if isinstance(flags, np.ndarray):
            flags = GSFlag(flags=flags, axes=("load", "pol", "time", "freq"))
        elif isinstance(flags, (str, Path)):
            flags = GSFlag.from_file(flags)

        flags._check_compat(self)

        if filt in self.flags:
            raise ValueError(f"Flags for filter '{filt}' already exist")

        new = self.update(flags={**self.flags, filt: flags})

        if append_to_file is None:
            append_to_file = new.filename is not None and new._file_appendable

        if append_to_file and (new.filename is None or not new._file_appendable):
            raise ValueError(
                "Cannot append to file without a filename specified on the object!"
            )

        if append_to_file:
            with h5py.File(new.filename, "a") as fl:
                try:
                    np.zeros(fl["data"]["data"].shape) * flags.full_rank_flags
                except ValueError:
                    # Can't append to file because it would be inconsistent.
                    return new

                flg_grp = fl["data"]["flags"]

                names_in_file = flg_grp.attrs.get("names", ())

                new_flags = tuple(k for k in new.flags if k not in names_in_file)

                for name in new_flags:
                    grp = flg_grp.create_group(name)
                    hickle.dump(new.flags[name], grp)

                flg_grp.attrs["names"] = tuple(new.flags.keys())

        return new

    def remove_flags(self, filt: str) -> GSData:
        """Remove flags for a given filter."""
        if filt not in self.flags:
            raise ValueError(f"No flags for filter '{filt}'")

        return self.update(flags={k: v for k, v in self.flags.items() if k != filt})

    def time_iter(self) -> Iterable[tuple[slice, slice, slice]]:
        """Return an iterator over the time axis of data-shape arrays."""
        for i in range(self.ntimes):
            yield (slice(None), slice(None), i, slice(None))

    def load_iter(self) -> Iterable[tuple[int]]:
        """Return an iterator over the load axis of data-shape arrays."""
        for i in range(self.nloads):
            yield (i,)

    def freq_iter(self) -> Iterable[tuple[slice, slice, slice]]:
        """Return an iterator over the frequency axis of data-shape arrays."""
        for i in range(self.nfreqs):
            yield (slice(None), slice(None), slice(None), i)
