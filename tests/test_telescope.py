"""Tests of the Telescope object."""

import attrs
import h5py
import numpy as np
import pytest
from astropy import units as un
from pygsdata import KNOWN_TELESCOPES, Telescope


@pytest.fixture()
def edgeslow():
    """Return the edges-low telescope."""
    return KNOWN_TELESCOPES["edges-low"]


def test_read_write_loop(tmp_path, edgeslow: Telescope):
    edgeslow.write(tmp_path / "edges-low.h5")
    new_edgeslow = Telescope.from_hdf5(tmp_path / "edges-low.h5")
    assert edgeslow == new_edgeslow


def test_read_write_loop_fileobj(tmp_path, edgeslow: Telescope):
    with h5py.File(tmp_path / "edges-low.h5", "w") as fl:
        edgeslow.write(fl)

    with h5py.File(tmp_path / "edges-low.h5", "r") as fl:
        new_edgeslow = Telescope.from_hdf5(fl)

    assert edgeslow == new_edgeslow


def test_read_write_loop_groupobj(tmp_path, edgeslow: Telescope):
    with h5py.File(tmp_path / "edges-low.h5", "w") as fl:
        telgroup = fl.create_group("telescope")
        edgeslow.write(telgroup)

    with h5py.File(tmp_path / "edges-low.h5", "r") as fl:
        telgroup = fl["telescope"]
        new_edgeslow = Telescope.from_hdf5(telgroup)

    assert edgeslow == new_edgeslow


def test_readwrite_bad_fname(tmp_path, edgeslow: Telescope):
    with pytest.raises(TypeError, match="Invalid type for fname"):
        Telescope.from_hdf5(fname=3)

    with pytest.raises(TypeError, match="Invalid type for fname"):
        edgeslow.write(fname=None)


def test_bad_pols(edgeslow):
    with pytest.raises(
        ValueError, match="Telescope must have at least one polarization"
    ):
        attrs.evolve(edgeslow, pols=())

    with pytest.raises(
        ValueError, match="Telescope must have 4 or fewer polarizations"
    ):
        attrs.evolve(edgeslow, pols=("XX", "XY", "YX", "YY", "pI"))

    with pytest.raises(ValueError, match="Invalid polarization: pZ"):
        attrs.evolve(edgeslow, pols=("XX", "pZ"))


def test_bad_integration_time(edgeslow):
    with pytest.raises(ValueError, match="Integration time must be positive"):
        attrs.evolve(edgeslow, integration_time=0 * un.s)

    with pytest.raises(ValueError, match="Integration time must be a scalar"):
        attrs.evolve(edgeslow, integration_time=np.array([13, 13]) * un.s)
