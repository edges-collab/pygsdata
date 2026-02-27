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


def test_calculate_rms_axis():
    """Test RMS calculation with axis parameter for 2D arrays."""
    arr = np.array([[1.0, 1.0], [1.0, 1.0], [3.0, 4.0]])
    rms_per_row = utils.calculate_rms(arr, axis=1)
    assert rms_per_row.shape == (3,)
    assert rms_per_row[0] == 1.0
    assert rms_per_row[1] == 1.0
    assert rms_per_row[2] == 3.536


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
