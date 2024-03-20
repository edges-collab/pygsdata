"""Tests of the GSFlag object."""
from itertools import product

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

    for ax, size in zip(axes, shape):
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


def test_any():
    f = GSFlag(
        flags=np.zeros((2, 3, 4, 5), dtype=bool), axes=("load", "pol", "time", "freq")
    )
    assert not f.any()

    f.flags[0] = True
    assert f.any()
    assert f.any(axis="load").all()
