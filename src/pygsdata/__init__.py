"""An interface for 21cm Global Signal Data."""

from pkg_resources import DistributionNotFound, get_distribution

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    __version__ = "unknown"
finally:
    del get_distribution, DistributionNotFound

from .gsdata import GSData
from .gsflag import GSFlag
from .history import History, Stamp
from .register import GSDATA_PROCESSORS, gsregister
from .select import select_freqs, select_lsts, select_times
