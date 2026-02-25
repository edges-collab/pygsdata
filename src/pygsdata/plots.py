"""Module providing standard plots for GSData objects."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from edges import modeling as mdl
from matplotlib.colors import Normalize, ScalarMappable

from .gsdata import GSData

mpl.use("agg")


def plot_waterfall(
    data: GSData,
    load: int = 0,
    pol: int = 0,
    which_flags: tuple[str] | None = None,
    ignore_flags: tuple[str] = (),
    ax: plt.Axes | None = None,
    cbar: bool = True,
    xlab: bool = True,
    ylab: bool = True,
    title: bool | str = True,
    attribute: str = "data",
    **imshow_kwargs,
):
    """Plot a waterfall from a GSData object.

    Parameters
    ----------
    data
        The GSData object to plot.
    load
        The index of the load to plot (only one load is plotted).
    pol
        The polarization to plot (only one polarization is plotted).
    which_flags
        A tuple of flag names to use in order to mask the data. If None, all flags are
        used. Send an empty tuple to ignore all flags.
    ignore_flags
        A tuple of flag names to ignore.
    ax
        The axis to plot on. If None, a new axis is created.
    cbar
        Whether to plot a colorbar.
    xlab
        Whether to plot an x-axis label.
    ylab
        Whether to plot a y-axis label.
    title
        Whether to plot a title. If True, the title is the year-day representation
        of the dataset. If a string, use that as the title.
    attribute
        The attribute to actually plot. Can be any attribute of the data object that has
        the same array shape as the primary data array. This includes "data",
        "residuals", "complete_flags", "nsamples".
    """
    q = getattr(data, attribute)
    if not hasattr(q, "shape") or q.shape != data.data.shape:
        raise ValueError(
            f"Cannot use attribute '{attribute}' as it doesn't have "
            "the same shape as data."
        )

    q = np.where(data.get_flagged_nsamples(which_flags, ignore_flags) == 0, np.nan, q)
    q = q[load, pol, :, :]

    if ax is None:
        ax = plt.subplots(1, 1, layout="constrained")[1]

    if attribute == "residuals":
        cmap = imshow_kwargs.pop("cmap", "coolwarm")
    else:
        cmap = imshow_kwargs.pop("cmap", "magma")

    times = data.times

    img = ax.imshow(
        q,
        origin="lower",
        extent=(
            data.freqs.min().to_value("MHz"),
            data.freqs.max().to_value("MHz"),
            times.jd.min(),
            times.jd.max(),
        ),
        cmap=cmap,
        aspect="auto",
        interpolation="none",
        **imshow_kwargs,
    )

    if xlab:
        ax.set_xlabel("Frequency [MHz]")
    if ylab:
        ax.set_ylabel("JD")

    dlst = data.times.jd[0, 0] * 24.0 - data.lsts.hourangle[0, 0]

    def jd2lst(jd):
        return jd * 24 - dlst  # spl_jd2lst(jd) % 24

    def lst2jd(lst):
        return lst + dlst

    v2 = ax.secondary_yaxis("right", functions=(jd2lst, lst2jd))
    v2.set_ylabel("LST [hour]")
    v2.yaxis.set_major_formatter(lambda x, pos: str(x % 24))

    if title and not isinstance(title, str):
        ax.set_title(f"{data.get_initial_yearday()}")
    if title and isinstance(title, str):
        ax.set_title(title)

    cb = plt.colorbar(img, ax=ax, pad=0.1) if cbar else None

    return ax, cb


def plot_rms_lst(
    data: GSData, 
    n_terms: int = 5, 
    offset: float = 0,
    **imshow_kwargs,
):
    """
    Creates two subplots: 
    top: freq (x-axis) vs residuals (y-axis) color-coded by LST (hr)
    bottom: LST (x-axis) vs RMS (y-axis)

    Parameters
    ----------
        data : GSData
            The data object containing frequency, LST, and residuals.
        n_terms : int
            The number of terms to use in the linlog model.
        offset : float
            The offset to add to the plot for each LST.
        imshow_kwargs : dict
            Keyword arguments to pass to the imshow function.

    """
    rms = []
    model_fit_freqs = data.freqs
    thermal_noise = get_thermal_noise(data)

    # Create subplots
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={"height_ratios": [2, 1]})

    # Plot residuals
    ax1 = axs[0]
    residuals = np.zeros((len(data.lsts), len(model_fit_freqs)))
    if data.residuals is not None:
        residuals = data.residual
    else:
        raise Warning(f"No residuals found in data object. Fitting {n_terms} term linlog model to data.")

        for i in range(len(data.lsts)):
            model = mdl.LinLog(n_terms=n_terms)
            linlog_model = model.at(x=model_fit_freqs)
            residuals[i, :] = linlog_model.fit(ydata=data.data[0, 0, i, :], xdata=model_fit_freqs).residual

        # Normalize LSTs for color mapping
        norm = Normalize(vmin=0, vmax=24)
        color = plt.cm.viridis(norm(data.lsts[i]))
        ax1.plot(
            model_fit_freqs,
            offset + residuals,
            color=color,
            label=str(data.lsts[i]),
        )

        # Append RMS
        rms.append(calculate_rms(residuals))

    ax1.set_xlabel("Frequency (MHz)")
    ax1.set_ylabel("Residuals (K)")
    ax1.set_title(f"{n_terms} term linlog, Averaged")
    ax1.grid()

    # Add colorbar to the first plot
    scalar_map = ScalarMappable(norm=norm, cmap="viridis")
    scalar_map.set_array([])
    cbar = fig.colorbar(scalar_map, ax=ax1)
    cbar.set_label("LST (hr)")

    # Plot RMS vs LST
    ax2 = axs[1]
    rms = np.array(rms)
    ax2.errorbar(
        data.lsts,
        rms,
        yerr=thermal_noise,
        marker="o",
        mfc="red",
        c="k",
        alpha=0.8,
    )
    ax2.set_ylabel("RMS (K)")
    ax2.set_xlabel("LST (hr)")
    ax2.grid()
    plt.tight_layout()

    return axs, cbar


def get_thermal_noise(gsdata_obj: GSData, n_terms=20):
    thermal_noise = []

    for i in range(len(gsdata_obj.lsts)):
        model = mdl.LinLog(n_terms=n_terms)
        model_fit_freqs = gsdata_obj.freqs

        LL = model.at(x=model_fit_freqs)
        res = LL.fit(ydata=gsdata_obj.data[0, 0, i, :], xdata=model_fit_freqs)

        thermal_noise.append(calculate_rms(res.residual))

    thermal_noise = np.array(thermal_noise)

    return thermal_noise


def calculate_rms(array, digits=3):
    rms = np.sqrt(np.mean(array**2))
    return round(rms, digits)
