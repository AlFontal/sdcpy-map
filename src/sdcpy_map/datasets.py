"""Dataset download and loading helpers for sdcpy-map."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from sdcpy_map.config import SDCMapConfig

PUBLIC_DATA_SOURCES = {
    "nino34_csv": "https://psl.noaa.gov/data/correlation/nina34.anom.csv",
    "ersstv5_nc": "https://raw.githubusercontent.com/pydata/xarray-data/master/ersstv5.nc",
    "coastline_zip": "https://naciscdn.org/naturalearth/110m/physical/ne_110m_coastline.zip",
}


def download_if_missing(url: str, destination: Path) -> Path:
    """Download a file only if it does not exist (or is empty)."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    urlretrieve(url, destination)
    return destination


def fetch_public_example_data(data_dir: Path | str) -> dict[str, Path]:
    """Fetch public demo inputs and return local file paths."""
    data_dir = Path(data_dir)
    out: dict[str, Path] = {}
    for key, url in PUBLIC_DATA_SOURCES.items():
        dest = data_dir / Path(url).name
        out[key] = download_if_missing(url, dest)
    return out


def load_driver_nino34(csv_path: Path | str, config: SDCMapConfig) -> pd.Series:
    """Load and clean Nino3.4 monthly anomalies for configured time range."""
    raw = pd.read_csv(csv_path)
    raw.columns = ["date", "nino34"]
    raw["date"] = pd.to_datetime(raw["date"])

    return (
        raw.loc[raw["nino34"] > -9990]
        .set_index("date")["nino34"]
        .sort_index()
        .loc[config.time_start : config.time_end]
    )


def load_sst_anomaly_subset(nc_path: Path | str, config: SDCMapConfig) -> xr.DataArray:
    """Load SST subset and build monthly anomalies in configured domain."""
    sst = xr.open_dataset(nc_path)["sst"]
    sst = sst.assign_coords(lon=(((sst.lon + 180) % 360) - 180)).sortby("lon")

    subset = sst.sel(
        time=slice(config.time_start, config.time_end),
        lat=slice(config.lat_max, config.lat_min),
        lon=slice(config.lon_min, config.lon_max),
    )

    subset_anom = subset.groupby("time.month") - subset.groupby("time.month").mean("time")

    return subset_anom.isel(
        lat=slice(None, None, config.lat_stride),
        lon=slice(None, None, config.lon_stride),
    )


def load_coastline(coastline_zip: Path | str) -> gpd.GeoDataFrame:
    """Load Natural Earth coastline geometry."""
    return gpd.read_file(coastline_zip)


def align_driver_to_field(driver: pd.Series, sst_anom: xr.DataArray) -> pd.Series:
    """Align driver index to SST anomaly timestamps."""
    idx = pd.DatetimeIndex(sst_anom.time.values)
    aligned = driver.reindex(idx)
    if aligned.isna().any():
        raise ValueError("Driver and SST time indexes do not align.")
    return aligned


def grid_coordinates(sst_anom: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    """Return lat/lon coordinate arrays as numpy arrays."""
    return sst_anom["lat"].to_numpy(), sst_anom["lon"].to_numpy()
