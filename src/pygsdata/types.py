"""Various types used throughout the package."""
from typing import Union

import astropy.units as un
from astropy.coordinates import Longitude

FreqType = un.Quantity["frequency"]
FreqRangeType = tuple[FreqType, FreqType]
LSTType = Union[un.Quantity[un.hourangle], Longitude]
LSTRangeType = tuple[LSTType, LSTType]
TimeType = un.Quantity["time"]
AngleType = un.Quantity["angle"]
