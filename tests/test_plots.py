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
