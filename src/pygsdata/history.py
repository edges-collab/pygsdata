"""Classes for defining the history of a GSData / GSFlag object."""

import contextlib
import datetime
import functools
import importlib
import inspect
import warnings
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, packages_distributions, version

import yaml
from attrs import asdict, define, evolve, field, fields
from attrs import validators as vld
from hickleable import hickleable

try:
    from typing import Self
except ImportError:
    from typing import Self


def _default_constructor(loader, tag_suffix, node):
    return f"{tag_suffix}: {node.value}"


yaml.add_multi_constructor("", _default_constructor, yaml.FullLoader)


@functools.cache
def _distribution_version(module: str) -> tuple[str, str] | None:
    """Return (distribution name, version) of the package that provides a module."""
    top = module.partition(".")[0]
    for dist in (*packages_distributions().get(top, ()), top):
        with contextlib.suppress(PackageNotFoundError):
            return dist, version(dist)
    return None


def _escape(text: str) -> str:
    """Escape square brackets so rich does not interpret them as markup."""
    return text.replace("[", r"\[")


def _resolve_qualname(qualname: str) -> object:
    """Import the object named by a fully-qualified ``module.qualname`` string."""
    parts = qualname.split(".")
    for i in range(len(parts) - 1, 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except ImportError:
            continue
        for attr in parts[i:]:
            obj = getattr(obj, attr)
        return obj
    raise ImportError(f"could not import any module from '{qualname}'")


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
    qualname
        The fully-qualified name (``module.qualname``) of the function that was
        applied. Together with ``versions``, this identifies the exact code that ran.
    description
        The summary line of the function's docstring at the time it was applied.
    """

    message: str = field(default="")
    function: str = field(default="")
    parameters: dict = field(factory=dict)
    versions: dict = field()
    timestamp: datetime.datetime = field(factory=datetime.datetime.now)
    qualname: str = field(default="")
    description: str = field(default="")

    @function.validator
    def _function_vld(self, _, value):
        if not value and not self.message:
            raise ValueError("History record must have a message or a function")

    @versions.default
    def _versions_default(self):
        out = {}
        for pkg in (
            "numpy",
            "astropy",
            "pygsdata",
        ):
            with contextlib.suppress(PackageNotFoundError):
                out[pkg] = version(pkg)
        return out

    @classmethod
    def from_function(
        cls, func: Callable, parameters: dict | None = None, message: str = ""
    ) -> Self:
        """Create a Stamp recording the application of a function.

        Records the function's name, fully-qualified name, docstring summary line,
        and the version of the package that provides it.
        """
        doc = inspect.getdoc(func) or ""
        stamp = cls(
            message=message,
            function=func.__name__,
            parameters=parameters or {},
            qualname=f"{func.__module__}.{func.__qualname__}",
            description=" ".join(doc.partition("\n\n")[0].split()),
        )
        if dist := _distribution_version(func.__module__):
            stamp = evolve(stamp, versions={**stamp.versions, dist[0]: dist[1]})
        return stamp

    def parameter_descriptions(self) -> dict[str, str]:
        """Return docstring descriptions of the recorded parameters.

        The descriptions are not stored in the history. They are read from the
        docstring of the function named by ``qualname``, which must be importable.
        If that function belongs to an installed package, the installed version must
        match the one recorded in ``versions``, so that the descriptions match the
        code that actually ran.

        Requires the optional ``docstring_parser`` package.

        Raises
        ------
        LookupError
            If the descriptions cannot be obtained. The message says why.
        """
        try:
            import docstring_parser
        except ImportError as e:
            raise LookupError("docstring_parser is not installed") from e

        if not self.qualname:
            raise LookupError("no qualname recorded")

        module = self.qualname.partition(".")[0]
        if module == "__main__":
            # The script that ran is not the one running now.
            raise LookupError(f"{self.qualname} was defined in a script")

        try:
            func = _resolve_qualname(self.qualname)
        except (ImportError, AttributeError) as e:
            raise LookupError(f"cannot import {self.qualname}") from e

        if dist := _distribution_version(module):
            name, installed = dist
            if (recorded := self.versions.get(name)) != installed:
                raise LookupError(
                    f"{name} version recorded as {recorded}, but {installed} is "
                    "installed"
                )

        try:
            doc = docstring_parser.parse(inspect.getdoc(func) or "")
        except docstring_parser.ParseError as e:
            raise LookupError(f"cannot parse docstring of {self.qualname}") from e

        return {
            p.arg_name: " ".join(p.description.split())
            for p in doc.params
            if p.arg_name in self.parameters and p.description
        }

    def __getstate__(self) -> dict:
        """Get the state for serialization."""
        return self._to_yaml_dict()

    def __setstate__(self, state: dict):
        """Set the state for deserialization."""
        # Go through from_yaml_dict so that fields missing from older files get
        # their defaults and unknown fields from newer files are dropped. hickle
        # adds 'item_index' when the stamp is an element of a container.
        state = {k: v for k, v in state.items() if k != "item_index"}
        self.__dict__.update(type(self).from_yaml_dict(state).__dict__)

    def _to_yaml_dict(self):
        dct = asdict(self)
        dct["timestamp"] = dct["timestamp"].isoformat()

        # For now, sanitize parameters that can't be represented in YAML by converting
        # them to strings. In the future, we may want to allow for more complex objects
        # to be represented in YAML.
        for k, v in dct["parameters"].items():
            try:
                yaml.load(yaml.dump(v), Loader=yaml.FullLoader)
            except Exception:  # noqa: BLE001
                # `v` can be an arbitrary user-supplied object, and yaml can raise
                # all sorts of errors (not just YAMLError) depending on its
                # __repr__/__reduce__ etc. Any failure here just means we can't
                # round-trip it through YAML, so fall back to a string repr.
                dct["parameters"][k] = str(v)

        return dct

    def __repr__(self):
        """Technical representation of the history record."""
        return yaml.dump(self._to_yaml_dict())

    def __str__(self):
        """Human-readable representation of the history record."""
        pstring = "\n        ".join(f"{k}: {v}" for k, v in self.parameters.items())
        vstring = " | ".join(f"{k} ({v})" for k, v in self.versions.items())

        return f"""{self.timestamp.isoformat()}
    function: {self.qualname or self.function}
    description: {self.description}
    message : {self.message}
    parameters:
        {pstring}
    versions: {vstring}
        """

    def pretty(self, annotate: bool = False):
        """Return a rich-compatible string representation of the history record.

        Parameters
        ----------
        annotate
            Whether to show the docstring description of each parameter next to its
            value. See :meth:`parameter_descriptions` for when these are available.
            Note that this imports the module named in ``qualname``.
        """
        descriptions = {}
        note = ""
        if annotate and self.parameters:
            try:
                descriptions = self.parameter_descriptions()
            except LookupError as e:
                note = f"\n        [dim italic](no descriptions: {_escape(str(e))})[/]"

        plines = []
        for k, v in self.parameters.items():
            line = f"[green]{k}[/]: [dim]{v}[/]"
            if k in descriptions:
                line += f"  [italic]# {_escape(descriptions[k])}[/]"
            plines.append(line)
        pstring = "\n        ".join(plines)
        vstring = " | ".join(f"{k} ([blue]{v}[/])" for k, v in self.versions.items())

        return f"""[bold underline blue]{self.timestamp.isoformat()}[/]
    [bold green]function[/]   : {self.qualname or self.function}
    [bold green]description[/]: {_escape(self.description)}
    [bold green]message [/]   : {self.message}
    [bold green]parameters[/] :{note}
        {pstring}
    [bold green]versions[/]   : {vstring}
        """

    @classmethod
    def from_repr(cls, repr_string: str):
        """Create a Stamp object from a string representation."""
        dct = yaml.load(repr_string, Loader=yaml.FullLoader)

        return cls.from_yaml_dict(dct)

    @classmethod
    def from_yaml_dict(cls, d: dict) -> Self:
        """Create a Stamp object from a dictionary representing a history record.

        Keys that are not fields of Stamp (e.g. written by a newer version of
        pygsdata) are dropped with a warning.
        """
        known = {f.name for f in fields(cls)}
        if unknown := sorted(set(d) - known):
            warnings.warn(
                f"Ignoring unknown history fields {unknown}. They may have been "
                "written by a newer version of pygsdata.",
                stacklevel=2,
            )
        d = {k: v for k, v in d.items() if k in known}
        if isinstance(d.get("timestamp"), str):
            d["timestamp"] = datetime.datetime.fromisoformat(d["timestamp"])
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
        self._keystring = tuple(stamp.timestamp.isoformat() for stamp in self.stamps)

    def __repr__(self):
        """Technical representation of the history."""
        out = tuple(s._to_yaml_dict() for s in self.stamps)
        return yaml.dump(out)

    def __str__(self):
        """Human-readable representation of the history."""
        return "\n\n".join(str(s) for s in self.stamps)

    def pretty(self, annotate: bool = False):
        """Return a rich-compatible string representation of the history.

        Parameters
        ----------
        annotate
            Whether to show docstring descriptions of each stamp's parameters. See
            :meth:`Stamp.pretty`.
        """
        return "\n\n".join(s.pretty(annotate=annotate) for s in self.stamps)

    def __getitem__(self, key):
        """Return the Stamp object corresponding to the given key."""
        if isinstance(key, int):
            return self.stamps[key]
        elif isinstance(key, str | datetime.datetime):
            if isinstance(key, datetime.datetime):
                key = key.isoformat()

            if key not in self._keystring:
                raise KeyError(
                    f"{key} not in history. Make sure the key is in ISO format."
                )
            return self.stamps[self._keystring.index(key)]
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
