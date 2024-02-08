"""Utilities for attrs used in GSData."""

from __future__ import annotations

import attrs
import numpy as np
from astropy import units as un
from astropy.coordinates import Longitude
from astropy.time import Time
from astropy.units import Quantity
from attrs import Attribute, cmp_using, field
from typing import Callable


def ndim_validator(ndim: int | tuple[int, ...]):
    """Validate that an array has a given number of dimensions."""
    if isinstance(ndim, int):
        ndim = (ndim,)

    def validator(inst, att, value):
        if value.ndim not in ndim:
            print(att.validator, value)
            raise ValueError(f"{att.name} must have ndim in {ndim}, got {value.ndim}")

    return validator


def shape_validator(shape: tuple[int | None, ...]):
    """Validate that an array has a given shape."""
    if isinstance(shape, int):
        shape = (shape,)

    def validator(inst, att, value):
        for i, (d0, d1) in enumerate(zip(shape, value.shape)):
            if d0 is not None and d0 != d1:
                raise ValueError(
                    f"Axis {i} of {att.name} must have size {d0}, got {d1}"
                )

    return validator


def array_converter(dtype=None, allow_none=False):
    """Construct an attrs converter to make arrays from other iterables."""

    def _converter(x: np.ArrayLike | Quantity | None):
        """Convert an array to a numpy array."""
        if x is None:
            if not allow_none:
                raise ValueError("None is not allowed")
            else:
                return None

        if not isinstance(x, (Quantity, Time, Longitude)):
            return np.asarray(x, dtype=dtype)
        else:
            return x

    return _converter


def unit_validator(unit):
    """Construct a validator to check that a Quantity has a given unit."""

    def validator(inst, att: Attribute, val: Quantity):
        if not isinstance(val, Quantity):
            raise TypeError(f"{att.name} must be a Quantity, got {type(val)}")

        if not val.unit.is_equivalent(unit):
            raise ValueError(
                f"{att.name} must have units compatible with {unit}, got {val.unit}"
            )

    return validator


def npfield(
    dtype=None,
    possible_ndims: tuple[int] | None = None,
    shape: tuple[int | None] | None = None,
    unit: un.Unit | None | str = None,
    validator: list | None | Callable = None,
    required: bool | None = None,
    **kwargs,
):
    """Construct an attrs field for a numpy array."""
    if kwargs.get("default", 1) is None and required is None:
        required = False
    elif required is None:
        required = True

    if dtype is bool:

        def cmp(x, y):
            if x is None and y is None:
                return True
            elif x is None or y is None:
                return False
            else:
                return np.array_equal(x, y)

    else:

        def cmp(x, y):
            if x is None and y is None:
                return True
            elif x is None or y is None:
                return False
            else:
                return x.shape == y.shape and np.allclose(x, y)

    if validator is None:
        validator = []
    elif callable(validator):
        validator = [validator]

    if possible_ndims is not None:
        validator.append(ndim_validator(possible_ndims))

    if shape is not None:
        validator.append(shape_validator(shape))

    if unit is not None:
        validator.append(unit_validator(unit))

    if validator:
        if required:
            kwargs["validator"] = validator
        else:
            kwargs["validator"] = attrs.validators.optional(validator)

    return field(
        eq=cmp_using(cmp),
        converter=array_converter(dtype=dtype, allow_none=not required),
        **kwargs,
    )


def timefield(
    possible_ndims: tuple[int] | None = None,
    shape: tuple[int] | None = None,
    validator=None,
    **kwargs,
):
    """Construct an attrs field for an astropy Time."""

    def cmp(x, y):
        if isinstance(x, Time) and isinstance(y, Time):
            return x.shape == y.shape and np.allclose(x.jd, y.jd)
        elif isinstance(x, Longitude) and isinstance(y, Longitude):
            return x.shape == y.shape and np.allclose(x.deg, y.deg)
        else:
            return False

    if validator is None:
        validator = []
    elif callable(validator):
        validator = [validator]

    if possible_ndims is not None:
        validator.append(ndim_validator(possible_ndims))

    if shape is not None:
        validator.append(shape_validator(shape))

    if validator:
        kwargs["validator"] = validator

    return field(eq=cmp_using(cmp), **kwargs)
