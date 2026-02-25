"""An interface for 21cm Global Signal Data."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pygsdata")
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"

__all__ = [
    "GSDATA_PROCESSORS",
    "KNOWN_TELESCOPES",
    "GSData",
    "GSFlag",
    "History",
    "Stamp",
    "Telescope",
    "gsdata_reader",
    "gsregister",
    "select_freqs",
    "select_lsts",
    "select_times",
]

from .constants import KNOWN_TELESCOPES
from .gsdata import GSData
from .gsflag import GSFlag
from .history import History, Stamp
from .readers import gsdata_reader
from .register import GSDATA_PROCESSORS, gsregister
from .select import select_freqs, select_lsts, select_times
from .telescope import Telescope
