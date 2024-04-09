"""Tests of the selectors."""
from pygsdata.select import select_loads


def test_select_loads(power_gsdata):
    """Test selecting loads."""
    gsd = select_loads(power_gsdata, loads=("ant",))
    assert gsd.nloads == 1

    gsd1 = select_loads(power_gsdata, indx=[0])
    assert gsd1.nloads == 1
