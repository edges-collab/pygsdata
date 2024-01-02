"""
A module containing the class GSData, a variant of UVData specific to single antennas.

The GSData object simplifies handling of radio astronomy data taken from a single
antenna, adding self-consistent metadata along with the data itself, and providing
key methods for data selection, I/O, and analysis.
"""
from __future__ import annotations

import astropy.units as un
import h5py
import hickle
import logging
import numpy as np
import warnings
from astropy.coordinates import EarthLocation, Longitude, UnknownSiteException
from astropy.time import Time
from attrs import converters as cnv
from attrs import define, evolve, field
from attrs import validators as vld
from collections.abc import Iterable
from functools import cached_property
from pathlib import Path
from read_acq.read_acq import ACQError
from typing import Literal

from . import coordinates as crd
from .attrs import npfield, timefield
from .constants import KNOWN_LOCATIONS
from .gsflag import GSFlag
from .history import History, Stamp

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
    freq_array
        The frequency array. This must be a 1D array of frequencies specified as an
        astropy Quantity.
    time_array
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
        "sample" of the data.
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

    data: np.ndarray = npfield(dtype=float, possible_ndims=(4,))
    freq_array: un.Quantity[un.MHz] = npfield(possible_ndims=(1,), unit=un.MHz)
    time_array: Time | Longitude = timefield(possible_ndims=(2,))
    telescope_location: EarthLocation = field(
        validator=vld.instance_of(EarthLocation),
        converter=lambda x: EarthLocation(*x)
        if not isinstance(x, EarthLocation)
        else x,
    )

    loads: tuple[str] = field(converter=tuple)
    nsamples: np.ndarray = npfield(dtype=float, possible_ndims=(4,))

    effective_integration_time: un.Quantity[un.s] = field(default=1 * un.s)
    flags: dict[str, GSFlag] = field(factory=dict)

    history: History = field(
        factory=History, validator=vld.instance_of(History), eq=False
    )
    telescope_name: str = field(default="unknown")
    residuals: np.ndarray | None = npfield(
        default=None, possible_ndims=(4,), dtype=float
    )

    data_unit: Literal[
        "power", "temperature", "uncalibrated", "uncalibrated_temp"
    ] = field(default="power")
    auxiliary_measurements: dict = field(factory=dict)
    time_ranges: Time | Longitude = timefield(shape=(None, None, 2))
    filename: Path | None = field(default=None, converter=cnv.optional(Path))
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

    @freq_array.validator
    def _freq_array_validator(self, attribute, value):
        if value.shape != (self.nfreqs,):
            raise ValueError(
                "freq_array must have the size nfreqs. "
                f"Got {value.shape} instead of {self.nfreqs}"
            )

    @time_array.validator
    def _time_array_validator(self, attribute, value):
        if value.shape != (self.ntimes, self.nloads):
            raise ValueError(
                f"time_array must have the size (ntimes, nloads), got {value.shape} "
                f"instead of {(self.ntimes, self.nloads)}"
            )

    @time_ranges.default
    def _time_ranges_default(self):
        if self.in_lst:
            return Longitude(
                np.concatenate(
                    (
                        self.time_array.hour[:, :, None],
                        self.time_array.hour[:, :, None]
                        + self.effective_integration_time.to_value("hour"),
                    ),
                    axis=-1,
                )
                * un.hour
            )
        else:
            return Time(
                np.concatenate(
                    (
                        self.time_array.jd[:, :, None],
                        self.time_array.jd[:, :, None]
                        + self.effective_integration_time.to_value("day"),
                    ),
                    axis=-1,
                ),
                format="jd",
            )

    @time_ranges.validator
    def _time_ranges_validator(self, attribute, value):
        if value.shape != (self.ntimes, self.nloads, 2):
            raise ValueError(
                f"time_ranges must have the size (ntimes, nloads, 2), got {value.shape}"
                f" instead of {(self.ntimes, self.nloads, 2)}."
            )

        if not self.in_lst and not np.all(value[..., 1] - value[..., 0] > 0):
            # TODO: properly check lst-type input, which can wrap...
            raise ValueError("time_ranges must all be greater than zero")

    @loads.default
    def _loads_default(self) -> tuple[str]:
        if self.data.shape[0] == 1:
            return ("ant",)
        elif self.data.shape[0] == 3:
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

    @effective_integration_time.validator
    def _effective_integration_time_validator(self, attribute, value):
        if not isinstance(value, un.Quantity):
            raise TypeError("effective_integration_time must be a Quantity")

        if not value.unit.is_equivalent("s"):
            raise ValueError("effective_integration_time must be in seconds")

    @auxiliary_measurements.validator
    def _aux_meas_vld(self, attribute, value):
        if not isinstance(value, dict):
            raise TypeError("auxiliary_measurements must be a dictionary")

        if isinstance(self.time_array, Longitude) and value:
            raise ValueError(
                "If times are LSTs, auxiliary_measurements cannot be specified"
            )

        for key, val in value.items():
            if not isinstance(key, str):
                raise TypeError("auxiliary_measurements keys must be strings")
            if not isinstance(val, np.ndarray):
                raise TypeError("auxiliary_measurements values must be arrays")
            if val.shape[0] != self.ntimes:
                raise ValueError(
                    "auxiliary_measurements values must have the size ntimes "
                    f"({self.ntimes}), but for {key} got shape {val.shape}"
                )

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

    @property
    def resids(self) -> np.ndarray | None:
        """The residuals of the data."""
        warnings.warn(
            DeprecationWarning("Use the 'residuals' attribute instead of 'resids'")
        )
        return self.residuals

    @classmethod
    def read_acq(
        cls,
        filename: str | Path,
        telescope_location: str | EarthLocation,
        name="{year}_{day}",
        **kw,
    ) -> GSData:
        """Read an ACQ file."""
        filename = Path(filename)

        try:
            from read_acq import read_acq
        except ImportError as e:
            raise ImportError(
                "read_acq is not installed -- install it to read ACQ files"
            ) from e

        _, (pant, pload, plns), anc = read_acq.decode_file(filename, meta=True)

        if pant.size == 0:
            raise ACQError(f"No data in file {filename}")

        times = Time(anc.data.pop("times"), format="yday", scale="utc")

        if isinstance(telescope_location, str):
            try:
                telescope_location = EarthLocation.of_site(telescope_location)
            except UnknownSiteException:
                try:
                    telescope_location = KNOWN_LOCATIONS[telescope_location]
                except KeyError:
                    raise ValueError(
                        "telescope_location must be an EarthLocation or a known site, "
                        f"got {telescope_location}"
                    )

        year, day, hour, minute = times[0, 0].to_value("yday", "date_hm").split(":")
        name = name.format(
            year=year, day=day, hour=hour, minute=minute, stem=filename.stem
        )
        return cls(
            data=np.array([pant.T, pload.T, plns.T])[:, np.newaxis],
            time_array=times,
            freq_array=anc.frequencies * un.MHz,
            data_unit="power",
            loads=("ant", "internal_load", "internal_load_plus_noise_source"),
            auxiliary_measurements={name: anc.data[name] for name in anc.data},
            filename=filename,
            telescope_location=telescope_location,
            name=name,
            **kw,
        )

    @classmethod
    def from_file(cls, filename: str | Path, **kw) -> GSData:
        """Create a GSData instance from a file.

        This method attempts to auto-detect the file type and read it.
        """
        filename = Path(filename)

        if filename.suffix == ".acq":
            return cls.read_acq(filename, **kw)
        elif filename.suffix == ".gsh5":
            return cls.read_gsh5(filename)
        else:
            raise ValueError("Unrecognized file type")

    @classmethod
    def read_gsh5(cls, filename: str) -> GSData:
        """Read a GSH5 file to construct the object."""
        with h5py.File(filename, "r") as fl:
            data = fl["data"][:]
            lat, lon, alt = fl["telescope_location"][:]
            telescope_location = EarthLocation(
                lat=lat * un.deg, lon=lon * un.deg, height=alt * un.m
            )
            times = fl["time_array"][:]

            if np.all(times < 24.0):
                time_array = Longitude(times * un.hour)
            else:
                time_array = Time(times, format="jd", location=telescope_location)
            freq_array = fl["freq_array"][:] * un.MHz
            data_unit = fl.attrs["data_unit"]
            objname = fl.attrs["name"]
            loads = fl.attrs["loads"].split("|")
            auxiliary_measurements = {
                name: fl["auxiliary_measurements"][name][:]
                for name in fl["auxiliary_measurements"].keys()
            }
            nsamples = fl["nsamples"][:]

            flg_grp = fl["flags"]
            flags = {}
            if "names" in flg_grp.attrs:
                flag_keys = flg_grp.attrs["names"]
                for name in flag_keys:
                    flags[name] = hickle.load(fl["flags"][name])

            filename = filename

            history = History.from_repr(fl.attrs["history"])

            if "residuals" in fl:
                residuals = fl["residuals"][()]
            else:
                residuals = None

        return cls(
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

    def write_gsh5(self, filename: str) -> GSData:
        """Write the data in the GSData object to a GSH5 file."""
        with h5py.File(filename, "w") as fl:
            fl["data"] = self.data
            fl["freq_array"] = self.freq_array.to_value("MHz")
            if self.in_lst:
                fl["time_array"] = self.time_array.hour
            else:
                fl["time_array"] = self.time_array.jd

            fl["telescope_location"] = np.array(
                [
                    self.telescope_location.lat.deg,
                    self.telescope_location.lon.deg,
                    self.telescope_location.height.to_value("m"),
                ]
            )

            fl.attrs["loads"] = "|".join(self.loads)
            fl["nsamples"] = self.nsamples
            fl.attrs[
                "effective_integration_time"
            ] = self.effective_integration_time.to_value("s")

            flg_grp = fl.create_group("flags")
            if self.flags:
                flg_grp.attrs["names"] = tuple(self.flags.keys())
                for name, flag in self.flags.items():
                    hickle.dump(flag, flg_grp.create_group(name))

            fl.attrs["telescope_name"] = self.telescope_name
            fl.attrs["data_unit"] = self.data_unit

            # Now history
            fl.attrs["history"] = repr(self.history)
            fl.attrs["name"] = self.name

            # Data model
            if self.residuals is not None:
                fl["residuals"] = self.residuals

            # Now aux measurements
            aux_grp = fl.create_group("auxiliary_measurements")
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

        if self.auxiliary_measurements or other.auxiliary_measurements:
            raise ValueError("Cannot add GSData objects with auxiliary measurements")

        if not np.allclose(self.freq_array, other.freq_array):
            raise ValueError("Cannot add GSData objects with different frequencies")

        if self.in_lst != other.in_lst:
            raise ValueError("Cannot add GSData objects with different time formats")

        if self.in_lst:
            if not np.allclose(self.time_array.hour, other.time_array.hour):
                raise ValueError("Cannot add GSData objects with different LSTs")
        else:
            if not np.allclose(self.time_array.jd, other.time_array.jd):
                raise ValueError("Cannot add GSData objects with different times")

        # If non of the above, then we have two GSData objects at the same times and
        # frequencies. Adding them should just be a weighted sum.
        nsamples = self.flagged_nsamples + other.flagged_nsamples
        d1 = np.ma.masked_array(self.data, mask=self.complete_flags)
        d2 = np.ma.masked_array(other.data, mask=other.complete_flags)

        mean = (self.flagged_nsamples * d1 + other.flagged_nsamples * d2) / nsamples

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
        )

    @cached_property
    def lst_array(self) -> Longitude:
        """The local sidereal time array."""
        if self.in_lst:
            return self.time_array
        else:
            return self.time_array.sidereal_time("apparent", self.telescope_location)

    @cached_property
    def lst_ranges(self) -> Longitude:
        """The local sidereal time array."""
        if self.in_lst:
            return self.time_ranges
        else:
            return self.time_ranges.sidereal_time("apparent", self.telescope_location)

    @cached_property
    def gha(self) -> np.ndarray:
        """The GHA's of the observations."""
        return Longitude(crd.lst2gha(self.lst_array.hour) * un.hourangle)

    def get_moon_azel(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the Moon's azimuth and elevation for each time in deg."""
        if self.in_lst:
            raise ValueError(
                "Cannot compute Moon positions when time array is not a Time object"
            )

        return crd.moon_azel(
            self.time_array[:, self.loads.index("ant")], self.telescope_location
        )

    def get_sun_azel(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the Sun's azimuth and elevation for each time in deg."""
        if self.in_lst:
            raise ValueError(
                "Cannot compute Sun positions when time array is not a Time object"
            )

        return crd.sun_azel(
            self.time_array[:, self.loads.index("ant")], self.telescope_location
        )

    def to_lsts(self) -> GSData:
        """
        Convert the time array to LST.

        Warning: this is an irreversible operation. You cannot go back to UTC after
        doing this. Furthermore, the auxiliary measurements will be lost.
        """
        if self.in_lst:
            return self

        return self.update(time_array=self.lst_array, auxiliary_measurements={})

    @property
    def in_lst(self) -> bool:
        """Returns True if the time array is in LST."""
        return isinstance(self.time_array, Longitude)

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

        if hours:
            subfmt = "date_hm"
        else:
            subfmt = "date"

        if self.in_lst:
            raise ValueError(
                "Cannot represent times as year-days, as the object is in LST mode"
            )

        out = self.time_array[0, self.loads.index("ant")].to_value("yday", subfmt)

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

        new = self.update(flags={**self.flags, **{filt: flags}})

        if append_to_file is None:
            append_to_file = new.filename is not None and new._file_appendable

        if append_to_file and (new.filename is None or not new._file_appendable):
            raise ValueError(
                "Cannot append to file without a filename specified on the object!"
            )

        if append_to_file:
            with h5py.File(new.filename, "a") as fl:
                try:
                    np.zeros(fl["data"].shape) * flags.full_rank_flags
                except ValueError:
                    # Can't append to file because it would be inconsistent.
                    return new

                flg_grp = fl["flags"]

                if "names" not in flg_grp.attrs:
                    names_in_file = ()
                else:
                    names_in_file = flg_grp.attrs["names"]

                new_flags = tuple(k for k in new.flags.keys() if k not in names_in_file)

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
