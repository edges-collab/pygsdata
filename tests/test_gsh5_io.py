"""Tests of the I/O for GSH5 data format."""

import attrs
import pytest

from pygsdata import GSData


@pytest.mark.parametrize(
    "data",
    [
        "simple_gsdata",
        "power_gsdata",
        "flagged_gsdata",
        "modelled_gsdata",
        "simple_gsdata_noaux",
    ],
)
def test_read_write_loop(data, request, tmp_path):
    """Test reading and writing a GSH5 file."""
    gsd = request.getfixturevalue(data)
    gsd.write_gsh5(tmp_path / "test.gsh5")
    new_gsd = GSData.from_file(tmp_path / "test.gsh5")

    flds = attrs.fields(GSData)
    for fld in flds:
        v1 = getattr(new_gsd, fld.name)
        v2 = getattr(gsd, fld.name)

        if not fld.eq:
            continue
        if fld.eq_key is not None:
            assert fld.eq_key(v1) == fld.eq_key(v2)
        else:
            assert v1 == v2

    assert gsd == new_gsd
