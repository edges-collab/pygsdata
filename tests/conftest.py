"""Top-level configuration for tests."""
import pytest
from mock_gsdata import mockgsd
from pygsdata import KNOWN_TELESCOPES, Telescope


@pytest.fixture(scope="session")
def edges() -> Telescope:
    return KNOWN_TELESCOPES["edges-low"]


@pytest.fixture(scope="session")
def simple_gsdata():
    return mockgsd()


@pytest.fixture(scope="session")
def power_gsdata():
    return mockgsd(as_power=True)
