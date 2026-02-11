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


def test_fetch_public_example_data_validates_keys(tmp_path: Path):
    with pytest.raises(ValueError):
        fetch_public_example_data(tmp_path, driver_key="missing", field_key=DEFAULT_FIELD_DATASET_KEY)
    with pytest.raises(ValueError):
        fetch_public_example_data(tmp_path, driver_key=DEFAULT_DRIVER_DATASET_KEY, field_key="missing")
