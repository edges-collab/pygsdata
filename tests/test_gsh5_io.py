"""Tests of the I/O for GSH5 data format."""

from unittest import mock

import attrs
import h5py
import numpy as np
import pytest
from astropy import units as un

from pygsdata import GSData, readers
from pygsdata.readers import GSH5ChecksumError
from pygsdata.utils import chunk_shape


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


def _flip_byte_in_chunk(path, dset_name: str):
    """Corrupt a file by flipping one byte in the first chunk of a dataset."""
    with h5py.File(path, "r") as fl:
        offset = fl[dset_name].id.get_chunk_info(0).byte_offset

    with path.open("r+b") as fl:
        fl.seek(offset)
        byte = fl.read(1)
        fl.seek(offset)
        fl.write(bytes([byte[0] ^ 0xFF]))


def _checksummed(path) -> dict[str, bool]:
    out = {}

    def _visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            out[name] = obj.fletcher32

    with h5py.File(path, "r") as fl:
        fl.visititems(_visit)
    return out


def test_write_checksums(flagged_gsdata, tmp_path):
    """Test that array datasets get checksums by default, and not otherwise."""
    flagged_gsdata.update(residuals=np.zeros(flagged_gsdata.data.shape)).write_gsh5(
        tmp_path / "test.gsh5"
    )
    checked = _checksummed(tmp_path / "test.gsh5")

    # Scalars can't be chunked, so can't carry a checksum.
    assert not checked["metadata/effective_integration_time"]

    for name in (
        "data/data",
        "data/nsamples",
        "data/residuals",
        "data/flags/hello/data/flags",
        "metadata/freqs",
        "metadata/times",
        "metadata/time_ranges",
        "metadata/lsts",
        "metadata/lst_ranges",
    ):
        assert checked[name], name

    flagged_gsdata.write_gsh5(tmp_path / "nocheck.gsh5", checksum=False)
    assert not any(_checksummed(tmp_path / "nocheck.gsh5").values())


def test_read_file_without_checksums(flagged_gsdata, tmp_path):
    """Files without checksums (e.g. written by older versions) still read."""
    flagged_gsdata.write_gsh5(tmp_path / "test.gsh5", checksum=False)
    assert GSData.from_file(tmp_path / "test.gsh5") == flagged_gsdata


def test_select_on_read_with_checksums(simple_gsdata, tmp_path):
    simple_gsdata.write_gsh5(tmp_path / "test.gsh5")
    new = GSData.from_file(
        tmp_path / "test.gsh5",
        selectors={"freq_selector": {"freq_range": (60 * un.MHz, 80 * un.MHz)}},
    )
    assert new.nfreqs < simple_gsdata.nfreqs


@pytest.mark.parametrize("dset", ["data/data", "data/flags/hello/data/flags"])
def test_corrupted_file_raises(flagged_gsdata, tmp_path, dset):
    """A corrupted checksummed dataset raises an error naming the dataset."""
    path = tmp_path / "test.gsh5"
    flagged_gsdata.write_gsh5(path)
    _flip_byte_in_chunk(path, dset)

    with pytest.raises(GSH5ChecksumError, match=f"/{dset}"):
        GSData.from_file(path)


def test_corrupted_file_without_checksums_reads_silently(simple_gsdata, tmp_path):
    """Documents why checksums exist: without them, corruption goes unnoticed."""
    path = tmp_path / "test.gsh5"
    simple_gsdata.write_gsh5(path, checksum=False)
    with h5py.File(path, "r") as fl:
        offset = fl["data/data"].id.get_offset()
    with path.open("r+b") as fl:
        fl.seek(offset)
        fl.write(b"\xff")

    assert not np.array_equal(GSData.from_file(path).data, simple_gsdata.data)


def test_other_oserrors_are_not_reported_as_corruption(simple_gsdata, tmp_path):
    path = tmp_path / "test.gsh5"
    simple_gsdata.write_gsh5(path)

    def _reader(fl, selectors):
        raise OSError("something else")

    with (
        mock.patch.object(readers._GSH5Readers, "v2", _reader),
        pytest.raises(OSError, match="something else") as exc,
    ):
        GSData.from_file(path)
    assert not isinstance(exc.value, GSH5ChecksumError)


def test_value_errors_do_not_trigger_scan(simple_gsdata, tmp_path):
    path = tmp_path / "test.gsh5"
    simple_gsdata.write_gsh5(path)

    def _reader(fl, selectors):
        raise ValueError("bad selector")

    with (
        mock.patch.object(readers._GSH5Readers, "v2", _reader),
        mock.patch.object(readers, "find_unreadable_datasets") as scan,
        pytest.raises(ValueError, match="bad selector"),
    ):
        GSData.from_file(path)
    scan.assert_not_called()


@pytest.mark.parametrize(
    ("shape", "itemsize", "target", "expected"),
    [
        ((3, 1, 1000, 16384), 8, 1024**2, (1, 1, 8, 16384)),
        ((1, 1, 10, 50), 8, 1024**2, (1, 1, 10, 50)),
        ((2, 4), 8, 16, (1, 2)),
        ((10,), 8, 16, (2,)),
        ((1, 1, 2, 1000), 8, 16, (1, 1, 1, 2)),
    ],
)
def test_chunk_shape(shape, itemsize, target, expected):
    assert chunk_shape(shape, itemsize, target) == expected
