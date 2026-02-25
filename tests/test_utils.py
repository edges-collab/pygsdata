"""Test the utils module."""

import numpy as np
from astropy.coordinates import Angle, Longitude

from pygsdata import utils


def test_calculate_rms():
    """Test RMS calculation for known arrays."""
    assert utils.calculate_rms(np.array([0.0])) == 0.0
    assert utils.calculate_rms(np.array([1.0, 1.0, 1.0, 1.0])) == 1.0
    assert utils.calculate_rms(np.array([3.0, 4.0])) == 3.536


def test_calculate_rms_digits():
    """Test that digits parameter controls rounding."""
    arr = np.array([1.0, 1.0, 1.0])
    assert utils.calculate_rms(arr, digits=0) == 1.0
    assert utils.calculate_rms(arr, digits=5) == 1.0
    assert utils.calculate_rms(np.array([1.0, 1.0, 2.0]), digits=2) == 1.41
    assert utils.calculate_rms(np.array([1.0, 1.0, 2.0]), digits=4) == 1.4142


def test_get_thermal_noise(simple_gsdata):
    """Test get_thermal_noise returns one value per LST."""
    thermal = utils.get_thermal_noise(simple_gsdata)
    assert thermal.shape == (len(simple_gsdata.lsts),)
    assert np.all(np.isfinite(thermal))
    assert np.all(thermal >= 0)


def test_get_thermal_noise_n_terms(simple_gsdata):
    """Test get_thermal_noise with custom n_terms."""
    thermal_5 = utils.get_thermal_noise(simple_gsdata, n_terms=5)
    thermal_20 = utils.get_thermal_noise(simple_gsdata, n_terms=20)
    expected_shape = (len(simple_gsdata.lsts),)
    assert thermal_5.shape == expected_shape
    assert thermal_20.shape == expected_shape
    assert np.all(np.isfinite(thermal_5))
    assert np.all(np.isfinite(thermal_20))


def test_angle_centre():
    zero = Angle(0, unit="hourangle")
    one = Angle(1, unit="hourangle")
    two = Angle(2, unit="hourangle")

    assert utils.angle_centre(zero, one) == Angle(0.5, unit="hourangle")
    assert utils.angle_centre(one, two) == Angle(1.5, unit="hourangle")
    assert utils.angle_centre(two, zero) == Angle(13.0, unit="hourangle")
    assert utils.angle_centre(Angle(23, unit="hourangle"), two) == Angle(
        24.5, unit="hourangle"
    )
    assert np.isclose(
        utils.angle_centre(Longitude(23, unit="hourangle"), Longitude(two)).hourangle,
        Longitude(0.5, unit="hourangle").hourangle,
    )

    assert utils.angle_centre(zero, one, p=0.75) == Angle(0.75, unit="hourangle")
