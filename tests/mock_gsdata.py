"""Create very simple mock GSData objects for testing purposes."""

from __future__ import annotations

import numpy as np
from astropy import units as un
from astropy.time import Time
from pygsdata import GSData, GSFlag
from pygsdata.constants import KNOWN_TELESCOPES


def data_constant(c: float):
    def inner(freqs, lsts, shape):
        return np.ones(shape) * c

    return inner


def data_powerlaw(freqs, lsts, shape):
    t75 = 4000.0 + np.sin(lsts.rad) * 1000.0
    specidx = -2.5
    data = np.ones(shape)
    return data * np.outer(t75, (freqs / (75 * un.MHz)) ** specidx)


def flag_constant(flagged: bool = False, axes: tuple[str] = ("time",)):
    def inner(shape):
        s = tuple(
            sh for ax, sh in zip(("load", "pol", "time", "freq"), shape) if ax in axes
        )
        flags = np.ones(s, dtype=bool) if flagged else np.zeros(s, dtype=bool)
        return GSFlag(flags=flags, axes=axes)

    return inner


def mockgsd(
    freq_range: tuple[un.Quantity[un.MHz], un.Quantity[un.MHz]] = (
        50 * un.MHz,
        100 * un.MHz,
    ),
    ntime: int = 10,
    nfreq: int = 50,
    npol: int = 1,
    time0: float = 2459900.27,
    dt: un.Quantity[un.s] = 40.0 * un.s,
    noise_level: float = 0.0,
    as_power: bool = False,
    data_creator: callable = data_constant(1.0),
    nsample_creator: callable = data_constant(1.0),
    flag_creators: dict[str, callable] | None = None,
    **kw,
):
    flag_creators = flag_creators or {}

    dshape = (1, npol, ntime, nfreq)
    shape = (3 if as_power else 1, npol, ntime, nfreq)

    dt = dt.to(un.day)
    freqs = (
        np.linspace(freq_range[0].value, freq_range[1].value, nfreq)
        * freq_range[0].unit
    )

    times = Time(
        np.arange(time0, (ntime - 0.1) * dt.value + time0, dt.value)[:, None],
        format="jd",
        location=KNOWN_TELESCOPES["edges-low"].location,
    )

    lsts = times.sidereal_time("apparent")

    skydata = data_creator(freqs, lsts, dshape)
    nsamples = nsample_creator(freqs, lsts, shape)
    flags = {key: creator(shape) for key, creator in flag_creators.items()}

    if noise_level > 0:
        rng = np.random.default_rng()
        skydata += rng.normal(0, skydata * noise_level)

    if as_power:
        p_load = 10 * np.ones_like(skydata)
        p_load_ns = 100 * np.ones_like(skydata)
        skydata = np.concatenate((skydata, p_load, p_load_ns), axis=0)
        times = Time(
            np.hstack(
                (
                    times.jd,
                    times.jd + dt.to_value("day") / 3,
                    times.jd + 2 * dt.to_value("day") / 3,
                )
            ),
            format="jd",
        )

    return GSData(
        data=skydata,
        freqs=freqs,
        times=times,
        telescope=KNOWN_TELESCOPES["edges-low"],
        nsamples=nsamples,
        flags=flags,
        effective_integration_time=dt / 3 if as_power else dt,
        data_unit="power" if as_power else "temperature",
        auxiliary_measurements={
            "ambient_hum": np.linspace(10.0, 90.0, ntime),
            "receiver_temp": np.linspace(10.0, 90.0, ntime),
        },
        **kw,
    )
