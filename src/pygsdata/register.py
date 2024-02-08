"""Register functions as processors for GSData objects."""

from __future__ import annotations

import attrs
import datetime
import functools
from typing import Callable, Literal

from .gsflag import GSFlag
from .pygsdata import GSData


class _Register:
    def __init__(self, func: Callable, kind: str) -> None:
        self.func = func
        self.kind = kind
        functools.update_wrapper(self, func, updated=())

    def __call__(
        self, data: GSData, *args, message: str = "", **kw
    ) -> GSData | list[GSData]:
        now = datetime.datetime.now()
        newdata = self.func(data, *args, **kw)

        history = {
            "message": message,
            "function": self.func.__name__,
            "parameters": kw,
            "timestamp": now,
        }

        kw = {"history": history}

        if self.kind not in ("supplement", "filter"):
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
                f"{self.func.__name__} returned {type(newdata)} "
                f"instead of GSData or list thereof."
            ) from e


GSDATA_PROCESSORS = {}

RegKind = Literal["gather", "calibrate", "filter", "reduce", "supplement"]


@attrs.define()
class gsregister:  # noqa: N801
    kind: RegKind = attrs.field(
        validator=attrs.validators.in_(
            ["gather", "calibrate", "filter", "reduce", "supplement"]
        )
    )

    def __call__(self, func: Callable) -> Callable:
        """Register a function as a processor for GSData objects."""
        out = _Register(func, self.kind)
        GSDATA_PROCESSORS[func.__name__] = out
        return out


# Some simple registered functions


@gsregister("supplement")
def add_flags(data: GSData, filt: str, flags: GSFlag) -> GSData:
    """Add flags to a GSData object."""
    return data.add_flags(filt, flags)
