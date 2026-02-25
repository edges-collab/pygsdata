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

def test_plot_rms_lst(simple_gsdata):
    axs, cbar = plots.plot_rms_lst(simple_gsdata)
    assert axs is not None
    assert len(axs) == 2
    assert cbar is not None
    assert axs[0].get_xlabel() == "Frequency (MHz)"
    assert axs[0].get_ylabel() == "Residuals (K)"
    assert axs[1].get_xlabel() == "LST (hr)"
    assert axs[1].get_ylabel() == "RMS (K)"


@pytest.mark.parametrize("n_terms,offset", [(3, 0), (5, 0.1)])
def test_plot_rms_lst_params(simple_gsdata, n_terms, offset):
    """Test plot_rms_lst with different n_terms and offset."""
    axs, cbar = plots.plot_rms_lst(simple_gsdata, n_terms=n_terms, offset=offset)
    assert len(axs) == 2
    assert axs[0].get_title() == f"{n_terms} term linlog, Averaged"