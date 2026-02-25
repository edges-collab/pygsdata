"""Tests of the from_file method."""

from pathlib import Path

import numpy as np
from astropy import units as un
from mock_gsdata import mockgsd


def test_gsh5_lst_selector(tmp_path: Path):
    """Test that the lst selector works when reading a GSH5 file."""
    gsd = mockgsd(
        dt=5 * un.min,
        ntime=12 * 24,  # 24 hours of data at 5 min cadence
    )
    gsd.write_gsh5(tmp_path / "test.gsh5")

    selectors = {"lst_selector": {"lst_range": (0, 12)}}
    new_gsd = gsd.from_file(tmp_path / "test.gsh5", selectors=selectors)

    assert gsd.ntimes // 2 - 1 <= len(new_gsd.lsts) <= gsd.ntimes // 2 + 1
    assert np.all((new_gsd.lsts.hourangle >= 0) & (new_gsd.lsts.hourangle < 12))
