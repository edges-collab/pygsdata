"""Test the utils module."""

import numpy as np
from astropy.coordinates import Angle, Longitude

from pygsdata import utils


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
