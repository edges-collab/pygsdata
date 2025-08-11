"""Test the register module."""

import pytest

from pygsdata import gsregister
from pygsdata.gsdata import GSData


@gsregister("calibrate")
def bad_gsfunc(data: GSData) -> GSData:
    return 3


def test_bad_gsfunc_return(simple_gsdata):
    with pytest.raises(TypeError, match="bad_gsfunc returned <class 'int'>"):
        bad_gsfunc(simple_gsdata)
