"""Create very simple mock GSData objects for testing purposes."""

import numpy as np
from astropy import units as un
from astropy.time import Time
from edges_analysis.const import edges_location
from edges_analysis.coordinates import lst2gha
from edges_analysis.data import DATA_PATH
from edges_analysis.pygsdata import GSData
from edges_cal.tools import FrequencyRange
from scipy.interpolate import interp1d

# To get "reasonable" data values, read the model of the haslam sky convolved with
# the 30x30m ground-plane beam that we have in our data folder. This exact model is
# used in the total_power_filter as a rough model to guide when to cut atrocious
# things out.
model = np.load(
    DATA_PATH / "Lowband_30mx30m_Haslam_2p5_20minlst_50_100.npy", allow_pickle=True
)
msky = model[0][:, 25]  # this is the 75 MHz slice
mgha = model[2]
spl75 = interp1d(mgha, msky)


def create_mock_edges_data(
    flow: un.Quantity[un.MHz] = 50 * un.MHz,
    fhigh: un.Quantity[un.MHz] = 100 * un.MHz,
    ntime: int = 100,
    time0: float = 2459900.27,
    dt: un.Quantity[un.s] = 40.0 * un.s,
    add_noise: bool = False,
    as_power: bool = False,
):
    """
    Create mock EDGES data.

    Parameters
    ----------
    flow : un.Quantity[un.MHz], optional
        The lower frequency edge of the frequency range (default: 50 MHz).
    fhigh : un.Quantity[un.MHz], optional
        The upper frequency edge of the frequency range (default: 100 MHz).
    ntime : int, optional
        The number of time samples (default: 100).
    time0 : float, optional
        The starting time in Julian Date (default: 2459900.27).
    dt : un.Quantity[un.s], optional
        The time step between samples in seconds (default: 40.0 s).
    add_noise : bool, optional
        Whether to add random noise to the data (default: False).
    as_power : bool, optional
        Whether to return the data as power or temperature (default: False).

    Returns
    -------
    GSData
        The mock EDGES data.

    Examples
    --------
    >>> data = create_mock_edges_data()
    """
    dt = dt.to(un.day)
    freqs = FrequencyRange.from_edges(f_low=flow, f_high=fhigh, keep_full=False)
    times = Time(
        np.arange(time0, (ntime - 0.1) * dt.value + time0, dt.value)[:, None],
        format="jd",
    )

    lsts = times.sidereal_time("apparent", longitude=edges_location.lon)
    gha = lst2gha(lsts.hour)[:, 0]

    skydata = spl75(gha)[:, None] * ((freqs.freq / (75 * un.MHz)) ** (-2.5))[None, :]

    if add_noise:
        rng = np.random.default_rng()
        skydata += rng.normal(0, 0.001, skydata.shape) * skydata

    data = skydata[None, None]

    if as_power:
        p1 = np.ones_like(data)
        p2 = np.ones_like(data)
        data = np.concatenate((data, p1, p2), axis=0)
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
        print("TIMESHAPE: ", times.shape)
    return GSData(
        data=data,
        freq_array=freqs.freq,
        time_array=times,
        telescope_location=edges_location,
        nsamples=np.ones_like(data),
        effective_integration_time=dt / 3,
        telescope_name="Mock-EDGES",
        data_unit="power" if as_power else "temperature",
        auxiliary_measurements={
            "ambient_hum": np.linspace(10.0, 90.0, ntime),
            "receiver_temp": np.linspace(10.0, 90.0, ntime),
        },
    )
