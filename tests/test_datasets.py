import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sdcpy_map.config import SDCMapConfig
from sdcpy_map.datasets import (
    DEFAULT_DRIVER_DATASET_KEY,
    DEFAULT_FIELD_DATASET_KEY,
    align_driver_to_field,
    download_if_missing,
    fetch_public_example_data,
    grid_coordinates,
    load_driver_series,
    load_field_anomaly_subset,
)


def test_load_driver_series_psl_table(tmp_path: Path):
    pdo_table = """1948 2025
1948 -0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 1.00 1.10 1.20
1949 -9.90 -0.20 -0.30 -0.40 -0.50 -0.60 -0.70 -0.80 -0.90 -1.00 -1.10 -1.20
"""
    path = tmp_path / "pdo.data"
    path.write_text(pdo_table, encoding="utf-8")

    config = SDCMapConfig(time_start="1948-01-01", time_end="1949-12-01")
    driver = load_driver_series(path, config=config, driver_key="pdo")

    assert isinstance(driver.index, pd.DatetimeIndex)
    assert pd.Timestamp("1948-01-01") in driver.index
    # Missing-value code -9.90 must be dropped.
    assert pd.Timestamp("1949-01-01") not in driver.index


def test_load_driver_series_nino34_csv(tmp_path: Path):
    csv = "date,value\n2015-10-01,2.0\n2015-11-01,2.5\n2015-12-01,2.3\n"
    path = tmp_path / "nino.csv"
    path.write_text(csv, encoding="utf-8")

    config = SDCMapConfig(time_start="2015-10-01", time_end="2015-12-01")
    driver = load_driver_series(path, config=config, driver_key="nino34")

    assert len(driver) == 3
    assert np.isclose(driver.loc["2015-11-01"], 2.5)


def test_load_field_anomaly_subset_wraps_longitude_and_subsets(tmp_path: Path):
    time = pd.date_range("2000-01-01", periods=24, freq="MS")
    lat = np.array([30.0, 20.0, 10.0, 0.0])  # descending
    lon = np.array([0.0, 90.0, 180.0, 270.0])

    # Build simple synthetic field with monthly cycle and spatial structure.
    vals = np.zeros((len(time), len(lat), len(lon)), dtype=float)
    for t in range(len(time)):
        vals[t] = t + lat[:, None] * 0.01 + lon[None, :] * 0.001

    ds = xr.Dataset(
        {"air": (("time", "lat", "lon"), vals)},
        coords={"time": time, "lat": lat, "lon": lon},
    )
    path = tmp_path / "air.nc"
    ds.to_netcdf(path)

    config = SDCMapConfig(
        time_start="2000-01-01",
        time_end="2001-12-01",
        lat_min=5,
        lat_max=25,
        lon_min=-180,
        lon_max=0,
    )
    field = load_field_anomaly_subset(path, config=config, field_key="ncep_air")
    lats, lons = grid_coordinates(field)

    assert field.dims == ("time", "lat", "lon")
    assert lats.min() >= 0
    assert lats.max() <= 20
    assert lons.min() >= -180
    assert lons.max() <= 0

    # Anomalies should have approximately zero monthly mean.
    month_mean = field.groupby("time.month").mean("time")
    assert np.allclose(month_mean.values, 0.0, atol=1e-10)


def test_align_driver_to_field_success_and_failure():
    idx = pd.date_range("2000-01-01", periods=6, freq="MS")
    driver = pd.Series(np.arange(6), index=idx)
    field = xr.DataArray(
        np.zeros((6, 2, 2), dtype=float),
        dims=("time", "lat", "lon"),
        coords={"time": idx, "lat": [0, 1], "lon": [10, 20]},
    )

    aligned = align_driver_to_field(driver, field)
    assert aligned.index.equals(idx)

    bad_driver = driver.iloc[:-1]
    with pytest.raises(ValueError):
        align_driver_to_field(bad_driver, field)


def test_existing_non_empty_file_returns_without_network_by_default(tmp_path: Path, monkeypatch):
    path = tmp_path / "dataset.bin"
    path.write_bytes(b"cached")

    def fail_urlopen(*args, **kwargs):
        raise AssertionError("urlopen should not be called for cache hits")

    def fail_run(*args, **kwargs):
        raise AssertionError("subprocess.run should not be called for cache hits")

    def fail_urlretrieve(*args, **kwargs):
        raise AssertionError("urlretrieve should not be called for cache hits")

    monkeypatch.setattr("sdcpy_map.datasets.urlopen", fail_urlopen)
    monkeypatch.setattr("sdcpy_map.datasets.subprocess.run", fail_run)
    monkeypatch.setattr("sdcpy_map.datasets.urlretrieve", fail_urlretrieve)

    out = download_if_missing("https://example.invalid/data.bin", path)
    assert out == path
    assert path.read_bytes() == b"cached"


def test_existing_file_with_verify_remote_true_checks_head(tmp_path: Path, monkeypatch):
    path = tmp_path / "dataset.bin"
    path.write_bytes(b"old")
    state = {"head_calls": 0, "downloads": 0}

    class _FakeHeadResponse:
        headers = {"Content-Length": "10"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(request, timeout=20):
        state["head_calls"] += 1
        assert request.get_method() == "HEAD"
        return _FakeHeadResponse()

    def fake_run(cmd, check):
        state["downloads"] += 1
        out_path = Path(cmd[cmd.index("-o") + 1])
        out_path.write_bytes(b"0123456789")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("sdcpy_map.datasets.urlopen", fake_urlopen)
    monkeypatch.setattr("sdcpy_map.datasets.subprocess.run", fake_run)

    out = download_if_missing("https://example.invalid/data.bin", path, verify_remote=True)
    assert out == path
    assert state["head_calls"] == 1
    assert state["downloads"] == 1
    assert path.stat().st_size == 10


def test_existing_file_with_verify_remote_and_unknown_size_uses_cache(tmp_path: Path, monkeypatch):
    path = tmp_path / "dataset.bin"
    path.write_bytes(b"cached")
    state = {"head_calls": 0}

    class _FakeHeadResponse:
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(request, timeout=20):
        state["head_calls"] += 1
        assert request.get_method() == "HEAD"
        return _FakeHeadResponse()

    def fail_run(*args, **kwargs):
        raise AssertionError("download should not run when Content-Length is unavailable")

    monkeypatch.setattr("sdcpy_map.datasets.urlopen", fake_urlopen)
    monkeypatch.setattr("sdcpy_map.datasets.subprocess.run", fail_run)

    out = download_if_missing("https://example.invalid/data.bin", path, verify_remote=True)
    assert out == path
    assert state["head_calls"] == 1
    assert path.read_bytes() == b"cached"


def test_missing_file_with_offline_true_raises(tmp_path: Path):
    path = tmp_path / "missing.bin"
    with pytest.raises(RuntimeError, match="Offline mode"):
        download_if_missing("https://example.invalid/data.bin", path, offline=True)


def test_refresh_true_forces_download(tmp_path: Path, monkeypatch):
    path = tmp_path / "dataset.bin"
    path.write_bytes(b"cached")
    calls = {"downloads": 0}

    def fake_run(cmd, check):
        calls["downloads"] += 1
        out_path = Path(cmd[cmd.index("-o") + 1])
        out_path.write_bytes(b"refreshed")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("sdcpy_map.datasets.subprocess.run", fake_run)

    out = download_if_missing("https://example.invalid/data.bin", path, refresh=True)
    assert out == path
    assert calls["downloads"] == 1
    assert path.read_bytes() == b"refreshed"


def test_missing_file_downloads_successfully(tmp_path: Path, monkeypatch):
    path = tmp_path / "downloaded.bin"
    calls = {"downloads": 0}

    def fake_run(cmd, check):
        calls["downloads"] += 1
        out_path = Path(cmd[cmd.index("-o") + 1])
        out_path.write_bytes(b"content")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("sdcpy_map.datasets.subprocess.run", fake_run)

    out = download_if_missing("https://example.invalid/data.bin", path)
    assert out == path
    assert calls["downloads"] == 1
    assert path.read_bytes() == b"content"


def test_fetch_public_example_data_validates_keys(tmp_path: Path):
    with pytest.raises(ValueError):
        fetch_public_example_data(tmp_path, driver_key="missing", field_key=DEFAULT_FIELD_DATASET_KEY)
    with pytest.raises(ValueError):
        fetch_public_example_data(tmp_path, driver_key=DEFAULT_DRIVER_DATASET_KEY, field_key="missing")
