"""Tests of additional utilities for the attrs library."""

import re

import attrs
import numpy as np
import pytest
from astropy import units as un
from astropy.table import QTable
from astropy.time import Time
from astropy.units import Quantity

from pygsdata import attrs as pga


def test_ndim_validator_single_possible_dim():
    @attrs.define
    class Test2D:
        data = attrs.field(validator=pga.ndim_validator(2))

    with pytest.raises(ValueError, match=re.escape("data must have ndim in (2,)")):
        Test2D(data=np.zeros(3))

    x = Test2D(data=np.zeros((3, 3)))
    assert x.data.shape == (3, 3)


def test_ndim_validator_multiple_dims():
    @attrs.define
    class Test1D2D:
        data = attrs.field(validator=pga.ndim_validator((1, 2)))

    with pytest.raises(ValueError, match=re.escape("data must have ndim in (1, 2)")):
        Test1D2D(data=np.zeros((3, 3, 3)))

    x = Test1D2D(data=np.zeros((3, 3)))
    assert x.data.shape == (3, 3)


def test_shape_validator_tuple():
    @attrs.define
    class TestShape:
        data = attrs.field(validator=pga.shape_validator((3, 3)))

    with pytest.raises(ValueError, match="Axis 1 of data must have size 3, got 4"):
        TestShape(data=np.zeros((3, 4)))

    x = TestShape(data=np.zeros((3, 3)))
    assert x.data.shape == (3, 3)


def test_shape_validator_int():
    @attrs.define
    class TestShape:
        data = attrs.field(validator=pga.shape_validator(3))

    with pytest.raises(ValueError, match=re.escape("data must have shape (3,)")):
        TestShape(data=np.zeros((3, 4)))

    x = TestShape(data=np.zeros(3))
    assert x.data.shape == (3,)


def test_shape_validator_none():
    @attrs.define
    class TestShape:
        data = attrs.field(validator=pga.shape_validator((3, None)))

    x = TestShape(data=np.zeros((3, 4)))
    assert x.data.shape == (3, 4)


def test_array_converter_default():
    @attrs.define
    class TestArray:
        data = attrs.field(converter=pga.array_converter())

    x = TestArray(data=[3])
    print(x)
    assert x.data == np.array(3)

    with pytest.raises(ValueError, match="None is not allowed"):
        x = TestArray(data=None)

    x = TestArray(data=Quantity(3, "m"))
    assert x.data == 3 * un.m


def test_array_converter_allow_nonw():
    @attrs.define
    class TestArray:
        data = attrs.field(converter=pga.array_converter(allow_none=True))

    x = TestArray(data=None)
    assert x.data is None


def test_array_converter_allow_int_dtype():
    @attrs.define
    class TestArray:
        data = attrs.field(converter=pga.array_converter(dtype=int))

    x = TestArray(data=3.0)
    assert x.data.dtype == int


def test_unit_validator():
    @attrs.define
    class TestUnit:
        data = attrs.field(validator=pga.unit_validator("m"))

    with pytest.raises(ValueError, match="data must have units compatible with m"):
        TestUnit(data=Quantity(3, "s"))

    x = TestUnit(data=Quantity(3, "m"))
    assert x.data == 3 * un.m

    with pytest.raises(TypeError, match="data must be a Quantity"):
        TestUnit(data=3)


class TestNpField:
    """Test the npfield function."""

    def test_ndims(self):
        """Test passing possible_ndims."""

        @attrs.define
        class Test:
            x = pga.npfield(possible_ndims=2)
            y = pga.npfield(possible_ndims=(1, 2))
            z = pga.npfield(possible_ndims=None)

        z = Test(x=np.zeros((3, 4)), y=np.zeros(3), z=np.zeros((1, 2, 3)))
        assert z.x.shape == (3, 4)
        assert z.y.shape == (3,)
        assert z.z.shape == (1, 2, 3)

        with pytest.raises(ValueError, match=re.escape("x must have ndim in (2,)")):
            Test(x=np.zeros(3), y=np.zeros(3), z=np.zeros((1, 2, 3)))

    def test_dtype(self):
        """Test passing dtype."""

        @attrs.define
        class Test:
            x = pga.npfield(dtype=int)
            y = pga.npfield(dtype=float)
            z = pga.npfield(dtype=None)

        z = Test(x=np.zeros(3), y=np.zeros(3), z=np.zeros(3))
        assert z.x.dtype == int
        assert z.y.dtype == float
        assert z.z.dtype == float

    def test_shape(self):
        """Test passing shape."""

        @attrs.define
        class Test:
            x = pga.npfield(shape=(3,))
            y = pga.npfield(shape=(3, 3))
            z = pga.npfield(shape=None)

        z = Test(x=np.zeros(3), y=np.zeros((3, 3)), z=np.zeros(3))
        assert z.x.shape == (3,)
        assert z.y.shape == (3, 3)
        assert z.z.shape == (3,)

        with pytest.raises(ValueError, match="Axis 0 of x must have size 3, got 4"):
            Test(x=np.zeros(4), y=np.zeros((3, 3)), z=np.zeros(3))

    def test_unit(self):
        """Test passing unit."""

        @attrs.define
        class Test:
            x = pga.npfield(unit="m")
            y = pga.npfield(unit=None)

        z = Test(x=Quantity(3, "m"), y=np.zeros(3))
        assert z.x == 3 * un.m
        assert z.y.shape == (3,)

        with pytest.raises(ValueError, match="x must have units compatible with m"):
            Test(x=Quantity(3, "s"), y=np.zeros(3))

        with pytest.raises(TypeError, match="x must be a Quantity"):
            Test(x=3, y=np.zeros(3))

    def test_extra_validator(self):
        """Test passing a single validator."""

        @attrs.define
        class Test:
            x = pga.npfield(possible_ndims=2, validator=pga.shape_validator((3, 3)))

        z = Test(x=np.zeros((3, 3)))
        assert z.x.shape == (3, 3)

        with pytest.raises(
            ValueError, match=re.escape("Axis 1 of x must have size 3, got 4")
        ):
            Test(x=np.zeros((3, 4)))

    def test_extra_validators(self):
        """Test passing multiple validators."""

        @attrs.define
        class Test:
            x = pga.npfield(
                possible_ndims=2,
                validator=[
                    pga.shape_validator((3, 3)),
                    attrs.validators.instance_of(np.ndarray),
                ],
            )

        z = Test(x=np.zeros((3, 3)))
        assert z.x.shape == (3, 3)

    def test_required(self):
        """Test the required parameter."""

        @attrs.define
        class Test:
            x = pga.npfield(possible_ndims=2, required=True)
            z = pga.npfield(possible_ndims=2, required=False)
            y = pga.npfield(default=np.array([3]), required=None)
            d = pga.npfield(default=None, required=None)

        with pytest.raises(TypeError, match="missing 2 required positional arguments"):
            Test()

        z = Test(x=np.zeros((3, 3)), z=None)
        assert z == z


class TestTimeField:
    """Test the timefield function."""

    def test_comparison(self):
        """Test the comparison methods."""

        @attrs.define(frozen=False)
        class Test:
            time = pga.timefield()

        x = Test(time=Time("2020-01-01T00:00:00"))
        assert x == x

        y = Test(time=Time("2020-01-01T00:00:10"))
        assert x != y

    def test_creation(self):
        """Test the creation of the field."""

        @attrs.define
        class Test:
            time = pga.lstfield()

        with pytest.raises(TypeError, match="Longitude instances require units"):
            Test(time=3)


def test_cmp_qtable():
    """Test the cmp_qtable function."""
    assert pga.cmp_qtable(None, None)
    assert not pga.cmp_qtable(None, 3)
    assert not pga.cmp_qtable(3, None)

    x = QTable({"a": [1]})
    y = QTable({"a": [2]})
    z = QTable({"b": [1]})

    assert not pga.cmp_qtable(x, y)
    assert not pga.cmp_qtable(x, z)
    assert not pga.cmp_qtable(y, z)
    assert pga.cmp_qtable(x, x)
    assert pga.cmp_qtable(y, y)
    assert pga.cmp_qtable(z, z)


def test_cmp_bool_array():
    """Test the _cmp_bool_array function."""
    assert pga._cmp_bool_array(None, None)
    assert not pga._cmp_bool_array(None, 3)
    assert not pga._cmp_bool_array(3, None)

    x = np.zeros(3, dtype=bool)
    y = np.ones(3, dtype=bool)
    z = np.zeros(4, dtype=bool)

    assert not pga._cmp_bool_array(x, y)
    assert not pga._cmp_bool_array(x, z)
    assert not pga._cmp_bool_array(y, z)
    assert pga._cmp_bool_array(x, x)
    assert pga._cmp_bool_array(y, y)
    assert pga._cmp_bool_array(z, z)


def test_cmp_num_array():
    """Test the _cmp_num_array function."""
    assert pga._cmp_num_array(None, None)
    assert not pga._cmp_num_array(None, 3)
    assert not pga._cmp_num_array(3, None)

    x = np.zeros(3)
    y = np.ones(3)
    z = np.zeros(4)

    assert not pga._cmp_num_array(x, y)
    assert not pga._cmp_num_array(x, z)
    assert not pga._cmp_num_array(y, z)
    assert pga._cmp_num_array(x, x)
    assert pga._cmp_num_array(y, y)
    assert pga._cmp_num_array(z, z)
