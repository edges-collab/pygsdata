"""Module for the Telescope class.

This defines a very simple Telescope class to hold telescope-related info.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import attrs
import h5py
import numpy as np
from astropy import units as un
from astropy.coordinates import Angle, EarthLocation
from attrs import field
from typing_extensions import Self

from .attrs import unit_validator
from .types import TimeType


def _pol_converter(pols: Sequence[str]) -> tuple[str]:
    pols = [
        pol[0].lower() + pol[1].upper() if "p" in pol.lower() else pol.upper()
        for pol in pols
    ]
    return tuple(pols)


@attrs.define(slots=False, kw_only=True)
class Telescope:
    """Class representing a telescope.

    Parameters
    ----------
    name
        The name of the telescope.
    location
        The location of the telescope.
    pols
        The polarizations that the telescope can measure. This should be all
        of the available polarizations that the telescope can measure, not just those
        that are in a particular observation. The polarizations should be given as a
        tuple of strings, where each string is one of "XX", "XY", "YX", "YY", "pI",
        "pQ", "pU", or "pV".
    integration_time
        The integration time of the telescope. This should be a scalar Quantity.
        In principle each observation made by a telescope may have different integration
        time, but this is the default value.
    x_orientation
        The orientation of the X polarization. This should be an Angle, and represents
        the orientation with respect to East. The default is 0 degrees, which means that
        the X polarization is aligned with East (the angle rotates towards North).
    """

    # This version indicates the version of the Telescope object, not this entire
    # package. The semantic meaning here is that the version is {major}.{minor}.
    # Minor version increases indicate simple bug-fixes that don't change the essential
    # file format. Major version increases indicate changes to the file format, but
    # can often still be backwards-compatible with some caveats.
    __version__ = "1.0"

    name: str = attrs.field(converter=str)
    location: EarthLocation = attrs.field(
        validator=attrs.validators.instance_of(EarthLocation)
    )
    pols: tuple[str] = field(converter=_pol_converter)
    integration_time: TimeType = field(default=1 * un.s, validator=unit_validator(un.s))
    x_orientation: Angle = field(
        default=0 * un.deg,
        converter=Angle,
    )

    @pols.validator
    def _pols_vld(self, _, value):
        if len(value) < 1:
            raise ValueError("Telescope must have at least one polarization")
        if len(value) > 4:
            raise ValueError("Telescope must have 4 or fewer polarizations")

        possible_pols = ("XX", "XY", "YX", "YY", "pI", "pQ", "pU", "pV")
        for pol in value:
            if pol.upper() not in possible_pols:
                raise ValueError(f"Invalid polarization: {pol}")

    @integration_time.validator
    def _integration_time_vld(self, _, value):
        if value.size != 1:
            raise ValueError("Integration time must be a scalar")

        if value.value <= 0:
            raise ValueError("Integration time must be positive")

    def write(self, fname: str | Path | h5py.File | h5py.Group):
        """Write the telescope to an HDF5 file."""
        if isinstance(fname, (str, Path)):
            with h5py.File(fname, "a") as fl:
                self.write(fl)

        elif isinstance(fname, (h5py.File, h5py.Group)):
            if not fname.name.endswith("/telescope"):
                grp = fname.create_group("telescope")
            else:
                grp = fname

            grp.attrs["version"] = self.__version__
            grp.attrs["name"] = self.name
            grp.attrs["integration_time"] = self.integration_time.to(un.s).value
            grp.attrs["x_orientation"] = self.x_orientation.to(un.deg).value
            grp["location"] = np.array(
                [x.to_value("m") for x in self.location.to_geocentric()]
            )
            grp["pols"] = self.pols
        else:
            raise TypeError(f"Invalid type for fname: {type(fname)}")

    @classmethod
    def from_hdf5(cls, fname: str | Path | h5py.File | h5py.Group) -> Self:
        """Read a telescope from an HDF5 file."""
        if isinstance(fname, (str, Path)):
            with h5py.File(fname, "r") as fl:
                return Telescope.from_hdf5(fl)

        elif isinstance(fname, (h5py.File, h5py.Group)):
            grp = fname["telescope"] if not fname.name.endswith("/telescope") else fname

            # Check the file-format version
            version = grp.attrs["version"]
            major = version.split(".")[0]
            reader = getattr(cls, f"_read_hdf5_version_{major}", None)
            if reader is None:
                raise ValueError(f"Unsupported file format version: {version}")

            return reader(grp)
        else:
            raise TypeError(f"Invalid type for fname: {type(fname)}")

    @classmethod
    def _read_hdf5_version_1(cls, grp: h5py.Group):
        return cls(
            name=grp.attrs["name"],
            location=EarthLocation.from_geocentric(*(grp["location"][()] * un.m)),
            pols=tuple(x.decode() for x in grp["pols"][:]),
            integration_time=grp.attrs["integration_time"] * un.s,
            x_orientation=grp.attrs["x_orientation"] * un.deg,
        )
