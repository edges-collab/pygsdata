"""Useful constants."""
from astropy import coordinates as apc
from astropy import units as apu

KNOWN_LOCATIONS = {
    "edges": apc.EarthLocation(lat=-26.714778 * apu.deg, lon=116.605528 * apu.deg),
    "alan-edges": apc.EarthLocation(lat=-26.7 * apu.deg, lon=116.5 * apu.deg),
}
