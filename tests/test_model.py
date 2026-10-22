"""Tests of the storage of a model of the data."""

import h5py
import numpy as np
import pytest
from astropy import units as un

from pygsdata import GSData
from pygsdata.concat import concat
from pygsdata.select import select_freqs, select_loads, select_times


@pytest.fixture
def model_and_nan_data(simple_gsdata):
    """Return a model, and data that are NaN in some channels."""
    rng = np.random.default_rng(1)
    model = simple_gsdata.data + rng.normal(size=simple_gsdata.data.shape)
    data = simple_gsdata.data.copy()
    data[..., 3:7] = np.nan
    return model, simple_gsdata.update(data=data, model=model)


def test_model_kept_where_data_is_nan(model_and_nan_data):
    model, gsd = model_and_nan_data
    np.testing.assert_array_equal(gsd.model, model)
    assert np.all(np.isnan(gsd.residuals[..., 3:7]))
    np.testing.assert_allclose(
        gsd.residuals[..., 7:], gsd.data[..., 7:] - model[..., 7:], rtol=0, atol=0
    )


def test_residuals_input_is_converted(simple_gsdata):
    resids = np.full(simple_gsdata.data.shape, 0.5)
    gsd = simple_gsdata.update(residuals=resids)
    np.testing.assert_array_equal(gsd.model, simple_gsdata.data - 0.5)
    np.testing.assert_allclose(gsd.residuals, 0.5)

    # Also in the constructor.
    new = GSData(
        data=simple_gsdata.data,
        freqs=simple_gsdata.freqs,
        times=simple_gsdata.times,
        telescope=simple_gsdata.telescope,
        residuals=resids,
    )
    np.testing.assert_array_equal(new.model, gsd.model)

    # The residuals are converted with the new data, if given.
    new_data = 2 * simple_gsdata.data
    gsd2 = simple_gsdata.update(data=new_data, residuals=resids)
    np.testing.assert_array_equal(gsd2.model, new_data - 0.5)

    assert simple_gsdata.update(residuals=None).model is None


def test_model_and_residuals_both_given(simple_gsdata):
    arr = np.zeros(simple_gsdata.data.shape)
    with pytest.raises(ValueError, match="Only one of model and residuals"):
        simple_gsdata.update(model=arr, residuals=arr)

    with pytest.raises(ValueError, match="Only one of model and residuals"):
        GSData(
            data=simple_gsdata.data,
            freqs=simple_gsdata.freqs,
            times=simple_gsdata.times,
            telescope=simple_gsdata.telescope,
            residuals=arr,
            model=arr,
        )


def test_bad_model_shape(simple_gsdata):
    with pytest.raises(ValueError, match="model must have the same shape as data"):
        simple_gsdata.update(model=np.zeros((2, 1, 50, 100)))


def test_update_data_keeps_model(model_and_nan_data):
    model, gsd = model_and_nan_data
    new = gsd.update(data=gsd.data + 1)
    np.testing.assert_array_equal(new.model, model)
    np.testing.assert_allclose(new.residuals, gsd.residuals + 1)


def test_evolve_does_not_reapply_residuals(simple_gsdata):
    gsd = simple_gsdata.update(residuals=np.zeros(simple_gsdata.data.shape))
    new = gsd.update(name="other")
    np.testing.assert_array_equal(new.model, gsd.model)


def test_model_equality(model_and_nan_data):
    _, gsd = model_and_nan_data
    assert gsd != gsd.update(model=gsd.model + 1)


def test_gsh5_round_trip_keeps_model(model_and_nan_data, tmp_path):
    model, gsd = model_and_nan_data
    gsd.write_gsh5(tmp_path / "test.gsh5")

    with h5py.File(tmp_path / "test.gsh5", "r") as fl:
        assert fl.attrs["version"] == "2.3"
        assert "model" in fl["data"]
        # Written for older readers.
        assert "residuals" in fl["data"]

    new = GSData.from_file(tmp_path / "test.gsh5")
    np.testing.assert_array_equal(new.model, model)
    np.testing.assert_array_equal(new.data, gsd.data)  # NaN compare equal here


def test_read_file_with_only_residuals(model_and_nan_data, tmp_path):
    """Files written before GSH5 2.3 store only the residuals."""
    _, gsd = model_and_nan_data
    gsd.write_gsh5(tmp_path / "test.gsh5")
    with h5py.File(tmp_path / "test.gsh5", "a") as fl:
        del fl["data/model"]
        fl.attrs["version"] = "2.2"

    new = GSData.from_file(tmp_path / "test.gsh5")
    # The model is recovered except where the data are NaN.
    np.testing.assert_allclose(new.model[..., 7:], gsd.model[..., 7:])
    assert np.all(np.isnan(new.model[..., 3:7]))


def test_read_with_selection(model_and_nan_data, tmp_path):
    model, gsd = model_and_nan_data
    gsd.write_gsh5(tmp_path / "test.gsh5")
    new = GSData.from_file(
        tmp_path / "test.gsh5",
        selectors={"freq_selector": {"freq_range": (60 * un.MHz, 80 * un.MHz)}},
    )
    mask = (gsd.freqs >= 60 * un.MHz) & (gsd.freqs <= 80 * un.MHz)
    np.testing.assert_array_equal(new.model, model[..., mask])


def test_selection_keeps_model(model_and_nan_data):
    model, gsd = model_and_nan_data

    sel = select_freqs(gsd, indx=np.arange(2, 10))
    np.testing.assert_array_equal(sel.model, model[..., 2:10])

    sel = select_times(gsd, indx=np.arange(3))
    np.testing.assert_array_equal(sel.model, model[:, :, :3])

    sel = select_loads(gsd, indx=np.array([0]))
    np.testing.assert_array_equal(sel.model, model[:1])


def test_concat_keeps_model(model_and_nan_data):
    model, gsd = model_and_nan_data
    a = select_times(gsd, indx=np.arange(3))
    b = select_times(gsd, indx=np.arange(3, gsd.ntimes))
    new = concat([a, b], axis="time")
    np.testing.assert_array_equal(new.model, model)
