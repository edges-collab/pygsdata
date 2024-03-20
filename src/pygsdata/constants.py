"""Useful constants."""

from astropy import coordinates as apc
from astropy import units as apu

from .telescope import Telescope

KNOWN_TELESCOPES = {
    "edges-low": Telescope(
        name="edges-low",
        location=apc.EarthLocation(lat=-26.714778 * apu.deg, lon=116.605528 * apu.deg),
        pols=("xx",),
        integration_time=13.0 * apu.s,
        x_orientation=0.0 * apu.deg,
    )
}

galactic_centre_lst = 17.76111111111111 * apu.hourangle  # hours
