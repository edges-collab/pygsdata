"""Tests of the GSFlag object."""

from itertools import product
from pathlib import Path

import numpy as np
import pytest

from pygsdata import GSFlag, History, Stamp


def test_invalid_axes():
    with pytest.raises(ValueError, match="Axes must be unique"):
        GSFlag(flags=np.zeros((2, 2), dtype=bool), axes=("time", "time"))

    with pytest.raises(ValueError, match="Axes must be a subset of"):
        GSFlag(flags=np.zeros((2, 2), dtype=bool), axes=("time", "bad"))

    with pytest.raises(ValueError, match="Axes must be in order"):
        GSFlag(flags=np.zeros((2, 2), dtype=bool), axes=("time", "load"))

    with pytest.raises(ValueError, match="Axes must be specified"):
        GSFlag(
            flags=np.zeros((2, 2), dtype=bool),
        )

    with pytest.raises(
        ValueError, match="Number of axes must match number of dimensions"
    ):
        GSFlag(flags=np.zeros((2, 2), dtype=bool), axes=("time",))


def test_invalid_flags():
    with pytest.raises(ValueError, match="Flag array must have at most 4 dimensions"):
        GSFlag(
            flags=np.zeros((2, 2, 2, 2, 2), dtype=bool),
            axes=("time", "load", "pol", "freq"),
        )


def test_none_axes():
    f = GSFlag(flags=np.zeros((2,), dtype=bool), axes=("load",))
    assert f.nfreqs is None
    assert f.ntimes is None
    assert f.npols is None
    assert f.nloads == 2

    f = GSFlag(flags=np.zeros((2, 3, 4), dtype=bool), axes=("pol", "time", "freq"))
    assert f.nfreqs == 4
    assert f.ntimes == 3
    assert f.npols == 2
    assert f.nloads is None


@pytest.mark.parametrize(
    "axes",
    [("time", "freq"), ("load", "pol"), ("load",), ("load", "pol", "time", "freq")],
)
def test_read_write_loop(tmp_path, axes):
    shape = tuple(range(2, 2 + len(axes)))
    f = GSFlag(flags=np.zeros(shape, dtype=bool), axes=axes)
    f.write_gsflag(tmp_path / "test.gsflag")
    new_f = GSFlag.from_file(tmp_path / "test.gsflag")
    assert f == new_f


def test_update():
    f = GSFlag(
        flags=np.zeros((2, 3, 4, 5), dtype=bool), axes=("load", "pol", "time", "freq")
    )

    new_f = f.update(history=Stamp(message="just tagging a point in time"))

    assert len(new_f.history) == len(f.history) + 1

    new_f = f.update(history=History([Stamp(message="just tagging a point in time")]))

    assert len(new_f.history) == len(f.history) + 1

    new_f = f.update()

    assert len(new_f.history) == len(f.history)

    new_f = f.update(history={"message": "hey there"})
    assert len(new_f.history) == len(f.history) + 1


@pytest.mark.parametrize(
    "axes", product(["load", None], ["pol", None], ["time", None], ["freq", None])
)
def test_full_rank_flags(axes):
    axes = [a for a in axes if a is not None]
    shape = tuple(range(2, 2 + len(axes)))
    f = GSFlag(flags=np.zeros(shape, dtype=bool), axes=axes)
    assert f.full_rank_flags.ndim == 4

    for ax, size in zip(axes, shape, strict=False):
        idx = f._axes.index(ax)
        assert f.full_rank_flags.shape[idx] == size


def test_compat():
    f1 = GSFlag(
        flags=np.zeros((2, 3, 4, 5), dtype=bool), axes=("load", "pol", "time", "freq")
    )

    f2 = GSFlag(flags=np.zeros((1,), dtype=bool), axes=("load",))

    with pytest.raises(ValueError, match="Objects have different nloads"):
        f1._check_compat(f2)

    f2 = GSFlag(flags=np.zeros((1,), dtype=bool), axes=("pol",))

    with pytest.raises(ValueError, match="Objects have different npols"):
        f1._check_compat(f2)

    f2 = GSFlag(flags=np.zeros((1,), dtype=bool), axes=("time",))

    with pytest.raises(ValueError, match="Objects have different ntimes"):
        f1._check_compat(f2)

    f2 = GSFlag(flags=np.zeros((1,), dtype=bool), axes=("freq",))

    with pytest.raises(ValueError, match="Objects have different nfreqs"):
        f1._check_compat(f2)


def test_or():
    f1 = GSFlag(
        flags=np.zeros((2, 3, 4, 5), dtype=bool), axes=("load", "pol", "time", "freq")
    )

    f2 = GSFlag(
        flags=np.zeros((2, 3, 4, 5), dtype=bool), axes=("load", "pol", "time", "freq")
    )

    f3 = f1 | f2

    assert not f3.flags.any()

    f2 = GSFlag(
        flags=np.ones((2, 3, 4, 5), dtype=bool), axes=("load", "pol", "time", "freq")
    )

    f3 = f1 | f2

    assert f3.flags.all()

    f2 = GSFlag(flags=np.ones((2,), dtype=bool), axes=("load",))
    f3 = f1 | f2
    assert f3.flags.all()


def test_or_badtype():
    f1 = GSFlag(
        flags=np.zeros((2, 3, 4, 5), dtype=bool), axes=("load", "pol", "time", "freq")
    )
    with pytest.raises(TypeError, match="can only 'or' GSFlag objects"):
        f1 | "not a GSFlag"


def test_and():
    f1 = GSFlag(
        flags=np.zeros((2, 3, 4, 5), dtype=bool), axes=("load", "pol", "time", "freq")
    )

    f2 = GSFlag(
        flags=np.zeros((2, 3, 4, 5), dtype=bool), axes=("load", "pol", "time", "freq")
    )

    f3 = f1 & f2

    assert not f3.flags.any()

    f2 = GSFlag(
        flags=np.ones((2, 3, 4, 5), dtype=bool), axes=("load", "pol", "time", "freq")
    )
    assert f2.flags.all()
    f3 = f1 & f2

    assert not f3.flags.any()

    f2 = GSFlag(flags=np.ones((2,), dtype=bool), axes=("load",))
    f3 = f1 & f2
    assert not f3.flags.any()


def test_and_badtype():
    f1 = GSFlag(
        flags=np.zeros((2, 3, 4, 5), dtype=bool), axes=("load", "pol", "time", "freq")
    )
    with pytest.raises(TypeError, match="can only 'and' GSFlag objects"):
        f1 & "not a GSFlag"


def test_select():
    f = GSFlag(
        flags=np.zeros((2, 3, 4, 5), dtype=bool), axes=("load", "pol", "time", "freq")
    )

    f2 = f.select(idx=np.array([0]), axis="load")
    assert f2.nloads == 1

    f2 = f.select(idx=np.array([0, 1]), axis="load")
    assert f2.nloads == 2

    f2 = f.select(idx=np.array([0, 1]), axis="pol")
    assert f2.npols == 2

    f2 = f.select(idx=np.array([0, 1]), axis="time")
    assert f2.ntimes == 2

    f2 = f.select(idx=np.array([0, 1]), axis="freq")
    assert f2.nfreqs == 2

    f2 = f.select(idx=slice(0, 1), axis="load", squeeze=True)
    assert f2.nloads is None

    f2 = f.select(idx=np.array([True, False]), axis="load", squeeze=True)
    assert f2.nloads is None


def test_select_bad_axis():
    f = GSFlag(
        flags=np.zeros((2, 3, 4, 5), dtype=bool), axes=("load", "pol", "time", "freq")
    )
    with pytest.raises(ValueError, match="Axis bad not recognized"):
        f.select(idx=np.array([0]), axis="bad")


def test_select_non_existent_axis():
    f = GSFlag(flags=np.zeros((2, 3, 4), dtype=bool), axes=("load", "pol", "time"))
    f2 = f.select(idx=np.array([0]), axis="freq")
    assert f2 is f


def test_any():
    f = GSFlag(
        flags=np.zeros((2, 3, 4, 5), dtype=bool), axes=("load", "pol", "time", "freq")
    )
    assert not f.any()

    f.flags[0] = True
    assert f.any()
    assert f.any(axis="load").all()


def test_bad_filetype():
    with pytest.raises(ValueError, match="Unrecognized file type"):
        GSFlag.from_file(Path("my.file"))

    with pytest.raises(ValueError, match="Unrecognized file type"):
        GSFlag.from_file(Path("my.gsflag"), filetype="unrecognized")


class TestOpOnAxis:
    """Test the op_on_axis functionality."""

    def setup_class(self):
        """Make a couple of GSFlag instances that we can use in tests."""
        self.f = GSFlag(
            flags=np.zeros((2, 3, 4, 5), dtype=bool),
            axes=("load", "pol", "time", "freq"),
        )
        self.g = GSFlag(
            flags=np.ones((2, 3, 4), dtype=bool), axes=("load", "pol", "time")
        )

    def test_bad_axis(self):
        """Test that a good error is raised when specified axis is bad."""
        with pytest.raises(ValueError, match="Axis bad not recognized"):
            self.f.op_on_axis(np.any, axis="bad")

    def test_non_existent_axis(self):
        """Check thatnothing is done when the axis doesn't exist in this data."""
        gg = self.g.op_on_axis(np.any, axis="freq")
        assert gg is self.g

    def test_reduce(self):
        """Test that the axes are reduced appropriately."""
        """Test that operations that reduce over an axis work."""
        f = self.f.op_on_axis(np.any, axis="freq")
        assert f.axes == self.f.axes[:-1]


class TestConcat:
    """Test the concat functionality."""

    def setup_class(self):
        """Set the class up."""
        self.f1 = GSFlag(
            flags=np.zeros((2, 4, 5), dtype=bool), axes=("load", "time", "freq")
        )
        self.f2 = GSFlag(
            flags=np.ones((2, 4, 5), dtype=bool), axes=("load", "time", "freq")
        )

    def test_bad_other(self):
        """Test that concatenation with something other than a GSFlag fails."""
        with pytest.raises(TypeError, match="can only concatenate GSFlag objects"):
            self.f1.concat(3, axis="freq")

    def test_non_iterable_concat(self):
        """Check concatting non-iterable GSFlag."""
        new = self.f1.concat(self.f2, axis="freq")
        assert new.nfreqs == self.f1.nfreqs + self.f2.nfreqs

    def test_iterable_concat(self):
        """Check concatenation of list."""
        new = self.f1.concat((self.f2, self.f2), axis="time")
        assert new.ntimes == self.f1.ntimes + 2 * self.f2.ntimes

    def test_bad_axis(self):
        """Test that bad axis raises exception."""
        with pytest.raises(ValueError, match="Axis bad not recognized"):
            self.f1.concat(self.f2, axis="bad")

    def test_non_existent_axis(self):
        """Check that non-existent axis raises exception."""
        with pytest.raises(
            ValueError, match="Axis pol not present in this GSFlag object"
        ):
            self.f1.concat(self.f2, axis="pol")
