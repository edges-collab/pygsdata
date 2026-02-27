"""Test the plots module."""

import pytest

from pygsdata import plots


@pytest.mark.parametrize("title", [None, "a title"])
@pytest.mark.parametrize("attribute", ["data", "nsamples", "flagged_nsamples"])
def test_plot_waterfall(title, simple_gsdata, attribute):
    """Test that plotting a waterfall doesn't crash."""
    plots.plot_waterfall(
        simple_gsdata,
        attribute=attribute,
        title=title,
    )


def test_plot_waterfall_bad_attribute(simple_gsdata):
    with pytest.raises(ValueError, match="Cannot use attribute"):
        plots.plot_waterfall(simple_gsdata, attribute="flags")


def test_plot_model_residuals_vs_lst(modelled_gsdata):
    """Test that plot_model_residuals_vs_lst produces expected subplots."""
    axs, cbar = plots.plot_model_residuals_vs_lst(modelled_gsdata)
    assert axs is not None
    assert len(axs) == 2
    assert cbar is not None
    assert axs[0].get_xlabel() == "Frequency (MHz)"
    assert axs[0].get_ylabel() == "Residuals (K)"
    assert axs[1].get_xlabel() == "LST (hr)"
    assert axs[1].get_ylabel() == "RMS (K)"


@pytest.mark.parametrize("offset", [0, 0.1])
def test_plot_model_residuals_vs_lst_params(modelled_gsdata, offset):
    """Test plot_model_residuals_vs_lst with different offset values."""
    axs, _ = plots.plot_model_residuals_vs_lst(modelled_gsdata, offset=offset)
    assert len(axs) == 2
