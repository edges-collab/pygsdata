"""Top-level configuration for tests."""

import numpy as np
import pytest
from mock_gsdata import mockgsd
from pygsdata import KNOWN_TELESCOPES, GSFlag, Telescope


@pytest.fixture(scope="session")
def edges() -> Telescope:
    return KNOWN_TELESCOPES["edges-low"]


@pytest.fixture(scope="session")
def simple_gsdata():
    return mockgsd()


@pytest.fixture(scope="session")
def power_gsdata():
    return mockgsd(as_power=True)


@pytest.fixture()
def flagged_gsdata(simple_gsdata):
    """Return a GSData object with some flags."""
    return simple_gsdata.update(
        flags={
            "hello": GSFlag(
                flags=np.zeros((simple_gsdata.ntimes,), dtype=bool), axes=("time",)
            ),
        }
    )


@pytest.fixture()
def modelled_gsdata(simple_gsdata):
    """Return a GSData object with some residuals."""
    return simple_gsdata.update(
        residuals=np.zeros(simple_gsdata.data.shape),
    )


@pytest.fixture()
def simple_gsdata_noaux(simple_gsdata):
    return simple_gsdata.update(auxiliary_measurements=None)
