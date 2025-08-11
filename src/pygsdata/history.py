"""Classes for defining the history of a GSData / GSFlag object."""

import contextlib
import datetime
import warnings
from importlib.metadata import PackageNotFoundError, version

import yaml
from attrs import asdict, define, evolve, field
from attrs import validators as vld
from hickleable import hickleable

try:
    from typing import Self
except ImportError:
    from typing import Self


def _default_constructor(loader, tag_suffix, node):
    return f"{tag_suffix}: {node.value}"


yaml.add_multi_constructor("", _default_constructor, yaml.FullLoader)


@hickleable()
@define(frozen=True, slots=False)
class Stamp:
    """Class representing a historical record of a process applying to an object.

    Parameters
    ----------
    message
        A message describing the process. Optional -- either this or the function
        must be defined.
    function
        The name of the function that was applied. Optional -- either this or the
        message must be defined.
    parameter(s)
        The parameters passed to the function. Optional -- if ``function`` is defined,
        this should be specified.
    versions
        A dictionary of the versions of the software used to perform the process.
        Created by default when the History is created.
    timestamp
        A datetime object corresponding to the time the process was performed.
        By default, this is set to the time that the Stamp object is created.
    """

    message: str = field(default="")
    function: str = field(default="")
    parameters: dict = field(factory=dict)
    versions: dict = field()
    timestamp: datetime.datetime = field(factory=datetime.datetime.now)

    @function.validator
    def _function_vld(self, _, value):
        if not value and not self.message:
            raise ValueError("History record must have a message or a function")

    @versions.default
    def _versions_default(self):
        out = {}
        for pkg in (
            "edges-cal",
            "edges-io",
            "edges-analysis",
            "read_acq",
            "numpy",
            "astropy",
            "pygsdata",
        ):
            with contextlib.suppress(PackageNotFoundError):
                out[pkg] = version(pkg)
        return out

    def _to_yaml_dict(self):
        dct = asdict(self)
        dct["timestamp"] = dct["timestamp"].isoformat()
        return dct

    def __repr__(self):
        """Technical representation of the history record."""
        return yaml.dump(self._to_yaml_dict())

    def __str__(self):
        """Human-readable representation of the history record."""
        pstring = "        ".join(f"{k}: {v}" for k, v in self.parameters.items())
        vstring = " | ".join(f"{k} ({v})" for k, v in self.versions.items())

        return f"""{self.timestamp.isoformat()}
    function: {self.function}
    message : {self.message}
    parameters:
        {pstring}
    versions: {vstring}
        """

    def pretty(self):
        """Return a rich-compatible string representation of the history record."""
        pstring = "        ".join(
            f"[green]{k}[/]: [dim]{v}[/]" for k, v in self.parameters.items()
        )
        vstring = " | ".join(f"{k} ([blue]{v}[/])" for k, v in self.versions.items())

        return f"""[bold underline blue]{self.timestamp.isoformat()}[/]
    [bold green]function[/]  : {self.function}
    [bold green]message [/]  : {self.message}
    [bold green]parameters[/]:
        {pstring}
    [bold green]versions[/]  : {vstring}
        """

    @classmethod
    def from_repr(cls, repr_string: str):
        """Create a Stamp object from a string representation."""
        dct = yaml.load(repr_string, Loader=yaml.FullLoader)

        return cls.from_yaml_dict(dct)

    @classmethod
    def from_yaml_dict(cls, d: dict) -> Self:
        """Create a Stamp object from a dictionary representing a history record."""
        try:
            d["timestamp"] = datetime.datetime.strptime(
                d["timestamp"], "%Y-%m-%dT%H:%M:%S.%f"
            )
        except ValueError:
            d["timestamp"] = datetime.datetime.strptime(
                d["timestamp"], "%Y-%m-%dT%H:%M:%S.%f%z"
            )
        return cls(**d)


@hickleable()
@define(slots=False)
class History:
    """A collection of Stamp objects defining the history."""

    stamps: tuple[Stamp] = field(
        factory=tuple,
        converter=tuple,
        validator=vld.deep_iterable(vld.instance_of(Stamp), vld.instance_of(tuple)),
    )

    def __attrs_post_init__(self):
        """Define the timestamps as keys."""
        self._keysdates = tuple(stamp.timestamp for stamp in self.stamps)
        self._keystring = tuple(stamp.timestamp.isoformat() for stamp in self.stamps)

    def __repr__(self):
        """Technical representation of the history."""
        out = tuple(s._to_yaml_dict() for s in self.stamps)
        return yaml.dump(out)

    def __str__(self):
        """Human-readable representation of the history."""
        return "\n\n".join(str(s) for s in self.stamps)

    def pretty(self):
        """Return a rich-compatible string representation of the history."""
        return "\n\n".join(s.pretty() for s in self.stamps)

    def __getitem__(self, key):
        """Return the Stamp object corresponding to the given key."""
        if isinstance(key, int):
            return self.stamps[key]
        elif isinstance(key, str):
            if key not in self._keystring:
                raise KeyError(
                    f"{key} not in history. Make sure the key is in ISO format."
                )
            return self.stamps[self._keystring.index(key)]
        elif isinstance(key, datetime.datetime):
            if key not in self._keysdates:
                raise KeyError(f"{key} not in history")
            return self.stamps[self._keysdates.index(key)]
        else:
            raise KeyError(
                f"{key} not a valid key. Must be int, ISO date string, or datetime."
            )

    @classmethod
    def from_repr(cls, repr_string: str):
        """Create a History object from a string representation."""
        try:
            d = yaml.load(repr_string, Loader=yaml.FullLoader)
        except yaml.constructor.ConstructorError as e:
            warnings.warn(
                (
                    f"History was not readable, with error message {e}. "
                    "Returning empty history."
                ),
                stacklevel=2,
            )

            return cls()
        if d := yaml.load(repr_string, Loader=yaml.FullLoader):
            return cls(stamps=[Stamp.from_yaml_dict(s) for s in d])
        else:
            return cls()

    def add(self, stamp: Stamp | dict | tuple[Stamp] | tuple[dict] | Self):
        """Add a stamp to the history."""
        if isinstance(stamp, dict):
            stamp = (Stamp(**stamp),)

        if isinstance(stamp, Stamp):
            return evolve(self, stamps=(*self.stamps, stamp))

        if all(isinstance(s, Stamp | dict) for s in stamp):
            a = self
            for s in stamp:
                a = a.add(s)
            return a

        raise TypeError("stamp must be a Stamp or a dictionary")

    def __len__(self):
        """Return the number of stamps."""
        return len(self.stamps)

    def __iter__(self):
        """Iterate over the stamps."""
        return iter(self.stamps)
