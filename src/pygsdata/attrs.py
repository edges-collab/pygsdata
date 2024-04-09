"""Utilities for attrs used in GSData."""

from __future__ import annotations

from typing import Callable

import attrs
import numpy as np
from astropy import units as un
from astropy.coordinates import Longitude
from astropy.time import Time
from astropy.units import Quantity
from attrs import Attribute, cmp_using, field


def ndim_validator(ndim: int | tuple[int, ...]):
    """Validate that an array has a given number of dimensions."""
    if isinstance(ndim, int):
        ndim = (ndim,)

    def validator(inst, att, value):
        if value.ndim not in ndim:
            raise ValueError(f"{att.name} must have ndim in {ndim}, got {value.ndim}")

    return validator


def shape_validator(shape: tuple[int | None, ...]):
    """Validate that an array has a given shape."""
    if isinstance(shape, int):
        shape = (shape,)

    def validator(inst, att, value):
        if len(shape) != len(value.shape):
            raise ValueError(f"{att.name} must have shape {shape}, got {value.shape}")

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


def _cmp_bool_array(x, y):
    if x is None and y is None:
        return True
    elif x is None or y is None:
        return False
    else:
        return np.array_equal(x, y)


def _cmp_num_array(x, y):
    if x is None and y is None:
        return True
    elif x is None or y is None:
        return False
    else:
        return (x.shape == y.shape or x.size == 1 or y.size == 1) and np.allclose(x, y)


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
        eq=cmp_using(_cmp_bool_array if dtype is bool else _cmp_num_array),
        converter=array_converter(dtype=dtype, allow_none=not required),
        **kwargs,
    )


def _astropy_subclass_field(
    cls,
    defining_attr,
    possible_ndims: tuple[int] | None = None,
    shape: tuple[int] | None = None,
    validator=None,
    **kwargs,
):
    def cmp(x, y):
        return x.shape == y.shape and np.allclose(
            getattr(x, defining_attr), getattr(y, defining_attr), rtol=0, atol=1e-8
        )

    if validator is None:
        validator = [attrs.validators.instance_of(cls)]
    elif callable(validator):
        validator = [validator, attrs.validators.instance_of(cls)]

    if possible_ndims is not None:
        validator.append(ndim_validator(possible_ndims))

    if shape is not None:
        validator.append(shape_validator(shape))

    kwargs["validator"] = validator

    return field(eq=cmp_using(cmp), **kwargs)


def timefield(
    possible_ndims: tuple[int] | None = None,
    shape: tuple[int] | None = None,
    validator=None,
    **kwargs,
):
    """Construct an attrs field for an astropy Time."""
    return _astropy_subclass_field(
        Time, "jd", possible_ndims, shape, validator, **kwargs
    )


def lstfield(
    possible_ndims: tuple[int] | None = None,
    shape: tuple[int] | None = None,
    validator=None,
    **kwargs,
):
    """Construct an attrs field for an astropy Time."""
    return _astropy_subclass_field(
        Longitude, "rad", possible_ndims, shape, validator, **kwargs
    )


def cmp_qtable(x, y):
    """Compare two QTable objects."""
    if x is None and y is None:
        return True
    elif x is None or y is None:
        return False
    else:
        if type(x) != type(y):
            return False
        if x.columns.keys() != y.columns.keys():
            return False

        for key in x.columns:
            if not np.all(x[key] == y[key]):
                return False

    return True
