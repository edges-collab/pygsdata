"""An object to hold flag information."""

from __future__ import annotations

from collections.abc import Sequence
from functools import cached_property
from pathlib import Path
from typing import Protocol

import hickle
import numpy as np
from attrs import converters as cnv
from attrs import define, evolve, field
from attrs import validators as vld
from hickleable import hickleable

from .attrs import npfield
from .history import History, Stamp

try:
    from typing import Self
except ImportError:
    from typing import Self


class _GSDataSized(Protocol):
    nloads: int | None
    ntimes: int | None
    npols: int | None
    nfreqs: int | None


@hickleable()
@define(slots=False)
class GSFlag:
    """A generic container for Global-Signal data.

    Parameters
    ----------
    flags
        The flags as a boolean array. The array may have up to 4 dimensions -- load,
        pol, time, and freq -- but need not have all of the dimensions.
    axes
        A tuple of strings specifying the axes of the data array. The possible axes are
        "load", "pol", "time", and "freq". They must be in that order, but not all
        must be present, only as many as flags has dimensions.
    history
        A tuple of dictionaries, each of which is a record of a previous processing
        step.
    filename
        The filename from which the data was read (if any). Used for writing additional
        data if more is added (eg. flags, data model).
    """

    _axes = ("load", "pol", "time", "freq")

    flags: np.ndarray = npfield(dtype=bool)
    axes: tuple[str] = field(converter=tuple)
    history: History = field(
        factory=History, validator=vld.instance_of(History), eq=False
    )
    filename: Path | None = field(default=None, converter=cnv.optional(Path))

    @flags.validator
    def _flags_vld(self, _, value):
        if value.ndim > 4:
            raise ValueError("Flag array must have at most 4 dimensions")

    @axes.validator
    def _axes_vld(self, _, value):
        if len(set(value)) != len(value):
            raise ValueError(f"Axes must be unique, got {value}")

        if len(value) != self.flags.ndim:
            raise ValueError(
                f"Number of axes must match number of dimensions in flags. "
                f"Got {len(value)} axes and {self.flags.ndim} dimensions"
            )

        if any(ax not in self._axes for ax in value):
            raise ValueError("Axes must be a subset of load, pol, time, freq")

        idx = [value.index(ax) for ax in self._axes if ax in value]

        if idx != sorted(idx):
            raise ValueError(f"Axes must be in order {self._axes}")

    @axes.default
    def _axes_default(self):
        if self.flags.ndim == 4:
            return self._axes
        else:
            raise ValueError(
                "Axes must be specified if flag array has fewer than 4 dims"
            )

    @cached_property
    def nfreqs(self) -> int | None:
        """The number of frequency channels."""
        if "freq" not in self.axes:
            return None

        return self.flags.shape[self.axes.index("freq")]

    @cached_property
    def nloads(self) -> int | None:
        """The number of loads."""
        return None if "load" not in self.axes else self.flags.shape[0]

    @property
    def ntimes(self) -> int | None:
        """The number of times."""
        if "time" not in self.axes:
            return None
        return self.flags.shape[self.axes.index("time")]

    @property
    def npols(self) -> int | None:
        """The number of polarizations."""
        return self.flags.shape[self.axes.index("pol")] if "pol" in self.axes else None

    @classmethod
    def from_file(cls, filename: str | Path, filetype: str | None = None, **kw) -> Self:
        """Create a GSFlag instance from a file.

        This method attempts to auto-detect the file type and read it.
        """
        filename = Path(filename)

        if filetype is None:
            filetype = filename.suffix[1:]  # Remove the leading dot

        if filetype.lower() == "gsflag":
            return cls.read_gsflag(filename)
        else:
            raise ValueError(f"Unrecognized file type: {filetype}")

    @classmethod
    def read_gsflag(cls, filename: str) -> Self:
        """Read a GSFlag file to create the object."""
        obj = hickle.load(filename)
        return obj.update(
            history=Stamp("Read GSFlag file", parameters={"filename": filename})
        )

    def write_gsflag(self, filename: str) -> Self:
        """Write the data in the GSData object to a GSH5 file."""
        new = self.update(
            history=Stamp("Wrote GSFlag file", parameters={"filename": filename})
        )
        hickle.dump(new, filename, mode="w")
        return new.update(filename=filename)

    def update(self, **kwargs) -> Self:
        """Return a new GSFlag object with updated attributes."""
        # If the user passes a single dictionary as history, append it.
        # Otherwise raise an error, unless it's not passed at all.
        history = kwargs.pop("history", None)
        history = self.history if history is None else self.history.add(history)

        return evolve(self, history=history, **kwargs)

    @property
    def full_rank_flags(self) -> np.ndarray:
        """Return a full-rank flag array."""
        flg = self.flags.copy()

        if "load" not in self.axes:
            flg = np.expand_dims(flg, axis=0)

        if "pol" not in self.axes:
            flg = np.expand_dims(flg, axis=1)

        if "time" not in self.axes:
            flg = np.expand_dims(flg, axis=2)

        if "freq" not in self.axes:
            flg = np.expand_dims(flg, axis=3)

        return flg

    def _check_compat(self, other: _GSDataSized) -> None:
        if (
            self.nloads is not None
            and other.nloads is not None
            and self.nloads != other.nloads
        ):
            raise ValueError(
                "Objects have different nloads. Got "
                f"this={self.nloads} and that={other.nloads}."
            )

        if (
            self.npols is not None
            and other.npols is not None
            and self.npols != other.npols
        ):
            raise ValueError(
                "Objects have different npols. Got "
                f"this={self.npols} and that={other.npols}"
            )

        if (
            self.ntimes is not None
            and other.ntimes is not None
            and self.ntimes != other.ntimes
        ):
            raise ValueError(
                "Objects have different ntimes. Got "
                f"this={self.ntimes} and that={other.ntimes}"
            )

        if (
            self.nfreqs is not None
            and other.nfreqs is not None
            and self.nfreqs != other.nfreqs
        ):
            raise ValueError(
                "Objects have different nfreqs. Got "
                f"this={self.nfreqs} and that={other.nfreqs}"
            )

    def __or__(self, other: GSFlag) -> Self:
        """Take the product of two GSFlag objects and return a new one."""
        if not isinstance(other, GSFlag):
            raise TypeError("can only 'or' GSFlag objects")

        self._check_compat(other)
        new_flags = np.squeeze(self.full_rank_flags | other.full_rank_flags)
        axes = tuple(ax for ax in self._axes if ax in self.axes + other.axes)

        return self.update(
            flags=new_flags,
            axes=axes,
            history=self.history.add(other.history).add(
                Stamp("Multiplied GSFlag objects")
            ),
            filename=None,
        )

    def __and__(self, other: GSFlag) -> Self:
        """Take the product of two GSFlag objects and return a new one."""
        if not isinstance(other, GSFlag):
            raise TypeError("can only 'and' GSFlag objects")

        self._check_compat(other)
        new_flags = np.squeeze(self.full_rank_flags & other.full_rank_flags)

        axes = tuple(ax for ax in self._axes if ax in self.axes + other.axes)

        return self.update(
            flags=new_flags,
            axes=axes,
            history=self.history.add(other.history).add(
                Stamp("Multiplied GSFlag objects")
            ),
            filename=None,
        )

    def select(self, idx: np.ndarray | slice, axis: str, squeeze: bool = False) -> Self:
        """Select a subset of the data along the given axis."""
        if axis not in self._axes:
            raise ValueError(f"Axis {axis} not recognized")

        if isinstance(idx, slice):
            idx = np.arange(*idx.indices(self.flags.shape[self.axes.index(axis)]))
        elif idx.dtype == bool:
            idx = np.where(idx)[0]

        # Do nothing if the axis is not present
        if axis not in self.axes:
            return self

        new_flags = self.flags.copy()
        new_flags = np.take(new_flags, idx, axis=self.axes.index(axis))

        if squeeze:
            axes = tuple(
                ax
                for i, ax in enumerate(self.axes)
                if ax != axis or new_flags.shape[i] > 1
            )
            new_flags = np.squeeze(new_flags)
        else:
            axes = self.axes

        history = self.history.add(
            Stamp("Selected subset of data", parameters={"axis": axis, "idx": idx})
        )
        return self.update(
            flags=new_flags,
            axes=axes,
            history=history,
            filename=None,
        )

    def op_on_axis(self, op, axis: str) -> Self:
        """Apply an operation along the given axis."""
        if axis not in ("load", "pol", "time", "freq"):
            raise ValueError(f"Axis {axis} not recognized")

        # Do nothing if the axis is not present
        if axis not in self.axes:
            return self

        new_flags = self.flags.copy()
        new_flags = op(new_flags, axis=self.axes.index(axis))

        axes = tuple(ax for ax in self.axes if ax != axis)

        return self.update(
            flags=new_flags,
            axes=axes,
            history=self.history.add(
                Stamp("Applied operation to data", parameters={"axis": axis, "op": op})
            ),
            filename=None,
        )

    def any(self, axis: str | None = None) -> bool | Self:
        """Return True if any of the flags are True."""
        return self.flags.any() if axis is None else self.op_on_axis(np.any, axis)

    def all(self, axis: str | None = None) -> bool | Self:
        """Return True if any of the flags are True."""
        return self.flags.all() if axis is None else self.op_on_axis(np.all, axis)

    def concat(self, others: GSFlag | Sequence[GSFlag], axis: str) -> GSFlag:
        """Get a new GSFlag by concatenating other flags to this one."""
        if not hasattr(others, "__len__"):
            others = [others]

        if not all(isinstance(o, GSFlag) for o in others):
            raise TypeError("can only concatenate GSFlag objects")

        if axis not in ("load", "pol", "time", "freq"):
            raise ValueError(f"Axis {axis} not recognized")

        if axis not in self.axes:
            raise ValueError(f"Axis {axis} not present in this GSFlag object")

        new_flags = np.concatenate(
            [self.flags] + [o.flags for o in others], axis=self.axes.index(axis)
        )
        return self.update(
            flags=new_flags,
            history=self.history.add(
                Stamp("Concatenated GSFlag objects", parameters={"axis": axis})
            ),
            filename=None,
        )
