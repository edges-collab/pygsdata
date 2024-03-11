"""Module providing standard plots for GSData objects."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .gsdata import GSData


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
        ax = plt.subplots(1, 1)[1]

    if attribute == "resids":
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
            0,
            (
                (times.max() - times.min()).hour
                if data.in_lst
                else (times.max() - times.min()).to_value("hour")
            ),
        ),
        cmap=cmap,
        aspect="auto",
        interpolation="none",
        **imshow_kwargs,
    )

    if xlab:
        ax.set_xlabel("Frequency [MHz]")
    if ylab:
        if data.in_lst:
            ax.set_ylabel("LST")
        else:
            ax.set_ylabel("Hours into Observation")

    if title and not isinstance(title, str) and not data.in_lst:
        ax.set_title(f"{data.get_initial_yearday()}. LST0={data.lsts[0][0]:.2f}")

    if cbar:
        cb = plt.colorbar(img, ax=ax)
        cb.set_label(data.loads[load])

    return ax
