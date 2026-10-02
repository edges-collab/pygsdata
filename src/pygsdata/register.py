"""Register functions as processors for GSData objects."""

import contextlib
import functools
import inspect
from collections.abc import Callable, Sequence
from typing import Literal, get_args, get_origin

import attrs

from .gsdata import GSData
from .gsflag import GSFlag
from .history import Stamp

GSDATA_PROCESSORS = {}

RegKind = Literal["gather", "calibrate", "filter", "reduce", "supplement"]


def _register(func: callable, kind: RegKind) -> callable:
    sig = inspect.signature(func)
    first_param = next(iter(sig.parameters.keys()))

    annotation = sig.parameters[first_param].annotation
    # Handle string annotations (forward references)
    if isinstance(annotation, str):
        with contextlib.suppress(Exception):
            annotation = eval(annotation, func.__globals__)
    allowed = False
    if annotation is GSData:
        allowed = True
    elif get_origin(annotation) in (list, Sequence):
        args = get_args(annotation)
        if args and args[0] is GSData:
            allowed = True
    if not allowed:
        raise TypeError(
            f"{func.__name__} must accept a GSData object or "
            "Sequence[GSData] as the first argument"
        )

    @functools.wraps(func)
    def wrapper(data: GSData, *args, message: str = "", **kw) -> GSData | list[GSData]:
        newdata = func(data, *args, **kw)

        # Record every parameter the function actually used, including positional
        # arguments and defaults, so the history is reproducible even if defaults
        # change in the future.
        bound = sig.bind(data, *args, **kw)
        bound.apply_defaults()
        parameters = {}
        for name, value in list(bound.arguments.items())[1:]:
            param_kind = sig.parameters[name].kind
            if param_kind is inspect.Parameter.VAR_KEYWORD:
                parameters.update(value)
            elif param_kind is inspect.Parameter.VAR_POSITIONAL:
                if value:
                    parameters[name] = list(value)
            else:
                parameters[name] = value

        history = Stamp.from_function(func, parameters=parameters, message=message)

        kw = {"history": history}

        if kind not in ("supplement", "filter"):
            # Any function that is not a supplement or filter is CHANGING data,
            # and should no longer be associated with the original file, in the sense
            # that new flags and data models should not be added to the file.
            kw["file_appendable"] = False

        if isinstance(newdata, GSData):
            return newdata.update(**kw)
        try:
            return [nd.update(**kw) for nd in newdata]
        except Exception as e:
            raise TypeError(
                f"{func.__name__} returned {type(newdata)} "
                f"instead of GSData or list thereof."
            ) from e

    GSDATA_PROCESSORS[func.__name__] = wrapper
    return wrapper


@attrs.define()
class gsregister:  # noqa: N801
    """Decorator to register a function as a processor for GSData objects."""

    kind: RegKind = attrs.field(
        validator=attrs.validators.in_(
            ["gather", "calibrate", "filter", "reduce", "supplement"]
        )
    )

    def __call__(self, func: Callable) -> Callable:
        """Register a function as a processor for GSData objects."""
        return _register(func, self.kind)


# Some simple registered functions
@gsregister("supplement")
def add_flags(data: GSData, filt: str, flags: GSFlag) -> GSData:
    """Add flags to a GSData object.

    Parameters
    ----------
    data
        The data to which to add the flags.
    filt
        The name under which to store the flags.
    flags
        The flags to add.
    """
    return data.add_flags(filt, flags)
