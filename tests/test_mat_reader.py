"""MAT format/container regressions using small coordinate-coded HSI cubes."""
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy.io import savemat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data_loader as dl
from inspect_mat import inspect_mat


def cube():
    return np.arange(12 * 10 * 7, dtype=np.float32).reshape(12, 10, 7)


def test_nested_v73_struct_and_all_axis_restoration(tmp_path):
    path = tmp_path / "Chikusei.mat"
    expected = cube()
    with h5py.File(path, "w") as handle:
        # The old loader only looked at top-level Dataset objects and failed.
        group = handle.create_group("chikusei")
        group.attrs["MATLAB_class"] = np.bytes_("struct")
        group.create_dataset("labels", data=np.zeros((12, 10), np.uint8))
        data = group.create_dataset("data", data=expected.transpose(2, 1, 0))
        data.attrs["MATLAB_class"] = np.bytes_("single")
    actual = dl.read_hsi_mat(str(path), ["chikusei", "data"], expected_bands=7)
    np.testing.assert_array_equal(actual, expected)


def test_v73_cell_reference_and_cycle(tmp_path):
    path = tmp_path / "cells.mat"
    expected = cube()
    with h5py.File(path, "w") as handle:
        refs = handle.create_group("#refs#")
        data = refs.create_dataset("cube", data=expected.transpose(2, 1, 0))
        data.attrs["MATLAB_class"] = np.bytes_("single")
        group = handle.create_group("wrapper")
        group["00_cycle"] = group
        cell = group.create_dataset("chikusei", (1, 2), dtype=h5py.ref_dtype)
        cell[0, 0] = h5py.Reference()
        cell[0, 1] = data.ref
    actual = dl.read_hsi_mat(str(path), ["other"], expected_bands=7)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("container", ["top", "struct", "cell"])
def test_classic_mat_containers(tmp_path, container):
    expected = cube()
    if container == "top":
        value = expected
    elif container == "struct":
        value = {"labels": np.zeros((12, 10)), "observation": {"data": expected}}
    else:
        value = np.empty((1, 2), dtype=object)
        value[0, 0] = np.zeros((12, 10))
        value[0, 1] = {"data": expected}
    path = tmp_path / "classic.mat"
    savemat(path, {"chikusei": value})
    actual = dl.read_hsi_mat(str(path), ["chikusei"], expected_bands=7)
    np.testing.assert_array_equal(actual, expected)


def test_hdf5_ignores_wrong_band_cube_without_materializing_it(tmp_path, monkeypatch):
    expected = cube()
    path = tmp_path / "generic.mat"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("chikusei", data=np.zeros((12, 10, 3)))
        handle.create_dataset("observations/data", data=expected.transpose(2, 0, 1))
    original = h5py.Dataset.__array__

    def guarded(dataset, *args, **kwargs):
        assert dataset.name != "/chikusei", "Wrong-band cube must be skipped using metadata"
        return original(dataset, *args, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "__array__", guarded)
    actual = dl.read_hsi_mat(str(path), ["chikusei"], expected_bands=7)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("hdf5", [False, True])
def test_2d_labels_and_flattened_matrix_produce_shape_diagnostics(tmp_path, hdf5):
    path = tmp_path / "labels.mat"
    values = {"ground_truth": np.zeros((12, 10)), "data": np.zeros((120, 7))}
    if hdf5:
        with h5py.File(path, "w") as handle:
            for name, value in values.items():
                handle.create_dataset(name, data=value)
    else:
        savemat(path, values)
    with pytest.raises(RuntimeError) as error:
        dl.read_hsi_mat(str(path), ["data"], expected_bands=7)
    message = str(error.value)
    assert "ground_truth" in message and "shape=(12, 10)" in message
    assert "shape=(120, 7)" in message and "7 spectral bands" in message
    assert "No automatic reshape" in message and "inspect_mat.py" in message


def test_corrupt_mat_keeps_backend_error_in_diagnostic(tmp_path):
    path = tmp_path / "broken.mat"
    path.write_bytes(b"not a MATLAB file")
    with pytest.raises(RuntimeError, match="Reader details") as error:
        dl.read_hsi_mat(str(path), ["chikusei"])
    assert "scipy:" in str(error.value)


def test_hdf5_inspector_only_reads_metadata(tmp_path, monkeypatch, capsys):
    path = tmp_path / "inspect.mat"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("nested/cube", data=cube())
    def forbid_data_read(*args, **kwargs):
        raise AssertionError("Inspector must not read cube data")
    monkeypatch.setattr(h5py.Dataset, "__array__", forbid_data_read)
    inspect_mat(path)
    output = capsys.readouterr().out
    assert "/nested/cube" in output and "shape=(12, 10, 7)" in output


def test_classic_inspector_reads_variable_metadata(tmp_path, capsys):
    path = tmp_path / "inspect_classic.mat"
    savemat(path, {"chikusei": cube()})
    inspect_mat(path)
    assert "chikusei: shape=(12, 10, 7)" in capsys.readouterr().out
