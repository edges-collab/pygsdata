"""An interface for 21cm Global Signal Data."""
from .gsdata import GSData
from .gsflag import GSFlag
from .history import History, Stamp
from .register import GSDATA_PROCESSORS, gsregister
from .select import select_freqs, select_lsts, select_times
