"""Dataset download and loading helpers for sdcpy-map."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen, urlretrieve

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from sdcpy_map.config import SDCMapConfig


@dataclass(frozen=True)
class DriverDatasetSpec:
    """Specification for a time-series driver dataset."""

    key: str
    url: str
    parser: str
    description: str


@dataclass(frozen=True)
class FieldDatasetSpec:
    """Specification for a gridded mapped-variable dataset."""

    key: str
    url: str
    variable: str
    description: str
    wrap_longitude: bool = True


COASTLINE_URL = "https://naciscdn.org/naturalearth/110m/physical/ne_110m_coastline.zip"

DRIVER_DATASETS: dict[str, DriverDatasetSpec] = {
    "pdo": DriverDatasetSpec(
        key="pdo",
        url="https://psl.noaa.gov/data/correlation/pdo.data",
        parser="psl_table",
        description="NOAA PSL Pacific Decadal Oscillation monthly index.",
    ),
    "nao": DriverDatasetSpec(
        key="nao",
        url="https://psl.noaa.gov/data/correlation/nao.data",
        parser="psl_table",
        description="NOAA PSL North Atlantic Oscillation monthly index.",
    ),
    "nino34": DriverDatasetSpec(
        key="nino34",
        url="https://psl.noaa.gov/data/correlation/nina34.anom.csv",
        parser="nino34_csv",
        description="NOAA PSL Niño3.4 monthly anomaly index.",
    ),
}

FIELD_DATASETS: dict[str, FieldDatasetSpec] = {
    "ncep_air": FieldDatasetSpec(
        key="ncep_air",
        url="https://psl.noaa.gov/thredds/fileServer/Datasets/ncep.reanalysis.derived/surface/air.mon.mean.nc",
        variable="air",
        description="NOAA NCEP/NCAR reanalysis monthly mean near-surface air temperature.",
        wrap_longitude=True,
    ),
    "ersstv5_sst": FieldDatasetSpec(
        key="ersstv5_sst",
        url="https://psl.noaa.gov/thredds/fileServer/Datasets/noaa.ersst.v5/sst.mnmean.nc",
        variable="sst",
        description="NOAA ERSSTv5 monthly SST.",
        wrap_longitude=True,
    ),
    "oisst_v2_sst": FieldDatasetSpec(
        key="oisst_v2_sst",
        url="https://psl.noaa.gov/thredds/fileServer/Datasets/noaa.oisst.v2/sst.mnmean.nc",
        variable="sst",
        description="NOAA OISSTv2 monthly SST (legacy prototype source).",
        wrap_longitude=True,
    ),
}

DEFAULT_DRIVER_DATASET_KEY = "pdo"
DEFAULT_FIELD_DATASET_KEY = "ncep_air"


def download_if_missing(
    url: str,
    destination: Path,
    *,
    refresh: bool = False,
    verify_remote: bool = False,
    offline: bool = False,
) -> Path:
    """Download a dataset with cache-first behavior and optional revalidation."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    def _is_non_empty(path: Path) -> bool:
        return path.exists() and path.stat().st_size > 0

    def _cleanup_zero_byte(path: Path) -> None:
        if path.exists() and path.stat().st_size == 0:
            path.unlink()

    local_ready = _is_non_empty(destination)
    if local_ready and not refresh:
        # Offline mode never performs remote checks; use local cache directly.
        if offline or not verify_remote:
            return destination

        try:
            request = Request(url, method="HEAD")
            with urlopen(request, timeout=20) as response:
                remote_size_raw = response.headers.get("Content-Length")
            remote_size = int(remote_size_raw) if remote_size_raw is not None else None
        except Exception:
            remote_size = None

        # Only redownload when the remote size is known and local is clearly stale.
        if remote_size is None or destination.stat().st_size >= remote_size:
            return destination

    if offline:
        raise RuntimeError(
            f"Offline mode is enabled and '{destination}' is unavailable or requires refresh."
        )

    curl_cmd = [
        "curl",
        "--http1.1",
        "-L",
        "--fail",
        "--retry",
        "8",
        "--retry-all-errors",
        "-C",
        "-",
        "-o",
        str(destination),
        url,
    ]
    try:
        for attempt in range(4):
            try:
                subprocess.run(curl_cmd, check=True)
                if _is_non_empty(destination):
                    return destination
                raise RuntimeError(f"Downloaded file is empty: '{destination}'.")
            except (subprocess.CalledProcessError, RuntimeError):
                _cleanup_zero_byte(destination)
                if attempt == 3:
                    raise
    except FileNotFoundError:
        # Fall back to urllib in environments without curl.
        pass

    # Lightweight retry loop for urllib downloads.
    attempts = 3
    for idx in range(attempts):
        try:
            urlretrieve(url, destination)
            if _is_non_empty(destination):
                return destination
            raise RuntimeError(f"Downloaded file is empty: '{destination}'.")
        except Exception:
            _cleanup_zero_byte(destination)
            if idx == attempts - 1:
                raise

    return destination


def fetch_public_example_data(
    data_dir: Path | str,
    driver_key: str = DEFAULT_DRIVER_DATASET_KEY,
    field_key: str = DEFAULT_FIELD_DATASET_KEY,
    include_coastline: bool = True,
) -> dict[str, Path]:
    """Fetch selected public demo inputs and return local file paths."""
    if driver_key not in DRIVER_DATASETS:
        supported = ", ".join(sorted(DRIVER_DATASETS))
        raise ValueError(f"Unknown driver dataset '{driver_key}'. Supported: {supported}.")
    if field_key not in FIELD_DATASETS:
        supported = ", ".join(sorted(FIELD_DATASETS))
        raise ValueError(f"Unknown field dataset '{field_key}'. Supported: {supported}.")

    data_dir = Path(data_dir)
    out: dict[str, Path] = {}

    driver_spec = DRIVER_DATASETS[driver_key]
    field_spec = FIELD_DATASETS[field_key]
    out["driver"] = download_if_missing(driver_spec.url, data_dir / Path(driver_spec.url).name)
    out["field"] = download_if_missing(field_spec.url, data_dir / Path(field_spec.url).name)

    if include_coastline:
        out["coastline"] = download_if_missing(COASTLINE_URL, data_dir / Path(COASTLINE_URL).name)

    return out


def _parse_psl_table_driver(path: Path | str) -> pd.Series:
    """Parse NOAA PSL monthly climate-index table format."""
    rows: list[tuple[pd.Timestamp, float]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 13:
                continue
            year_token = parts[0]
            if not year_token.lstrip("-").isdigit():
                continue

            year = int(year_token)
            for month_idx, raw in enumerate(parts[1:13], start=1):
                value = float(raw)
                # Missing-value codes used by PSL index tables.
                if np.isclose(value, -9.90) or np.isclose(value, -99.90):
                    continue
                rows.append((pd.Timestamp(year=year, month=month_idx, day=1), value))

    if not rows:
        raise ValueError(f"No valid monthly rows were found in '{path}'.")

    driver = pd.Series(
        [value for _, value in rows],
        index=pd.DatetimeIndex([date for date, _ in rows]),
        name="driver",
    ).sort_index()
    return driver


def _parse_nino34_csv_driver(path: Path | str) -> pd.Series:
    """Parse NOAA PSL Niño3.4 CSV format."""
    raw = pd.read_csv(path)
    raw.columns = ["date", "value"]
    raw["date"] = pd.to_datetime(raw["date"])
    return (
        raw.loc[raw["value"] > -9990]
        .set_index("date")["value"]
        .sort_index()
        .rename("driver")
    )


def load_driver_series(
    driver_path: Path | str,
    config: SDCMapConfig,
    driver_key: str = DEFAULT_DRIVER_DATASET_KEY,
) -> pd.Series:
    """Load and clean a driver time-series for the configured period."""
    if driver_key not in DRIVER_DATASETS:
        supported = ", ".join(sorted(DRIVER_DATASETS))
        raise ValueError(f"Unknown driver dataset '{driver_key}'. Supported: {supported}.")

    parser = DRIVER_DATASETS[driver_key].parser
    if parser == "psl_table":
        driver = _parse_psl_table_driver(driver_path)
    elif parser == "nino34_csv":
        driver = _parse_nino34_csv_driver(driver_path)
    else:
        raise ValueError(f"Unsupported parser '{parser}' for driver dataset '{driver_key}'.")

    return driver.loc[config.time_start : config.time_end]


def _slice_for_lat(coord: xr.DataArray, lat_min: float, lat_max: float) -> slice:
    """Build latitude slice honoring coordinate order."""
    ascending = bool(coord.values[0] < coord.values[-1])
    if ascending:
        return slice(lat_min, lat_max)
    return slice(lat_max, lat_min)


def load_field_anomaly_subset(
    field_path: Path | str,
    config: SDCMapConfig,
    field_key: str = DEFAULT_FIELD_DATASET_KEY,
) -> xr.DataArray:
    """Load mapped variable subset and compute monthly anomalies."""
    if field_key not in FIELD_DATASETS:
        supported = ", ".join(sorted(FIELD_DATASETS))
        raise ValueError(f"Unknown field dataset '{field_key}'. Supported: {supported}.")

    spec = FIELD_DATASETS[field_key]
    ds = xr.open_dataset(field_path)
    if spec.variable not in ds.data_vars:
        available = ", ".join(ds.data_vars)
        raise ValueError(
            f"Variable '{spec.variable}' not found in '{field_path}'. Available: {available}."
        )
    field = ds[spec.variable]

    if spec.wrap_longitude and "lon" in field.coords:
        field = field.assign_coords(lon=(((field.lon + 180) % 360) - 180)).sortby("lon")

    subset = field.sel(
        time=slice(config.time_start, config.time_end),
        lat=_slice_for_lat(field["lat"], config.lat_min, config.lat_max),
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


def align_driver_to_field(driver: pd.Series, mapped_field: xr.DataArray) -> pd.Series:
    """Align a driver time series to mapped-field timestamps."""
    idx = pd.DatetimeIndex(mapped_field.time.values)
    aligned = driver.reindex(idx)
    if aligned.isna().any():
        raise ValueError("Driver and mapped-variable time indexes do not align.")
    return aligned


def grid_coordinates(mapped_field: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    """Return latitude/longitude coordinate arrays as NumPy arrays."""
    return mapped_field["lat"].to_numpy(), mapped_field["lon"].to_numpy()


# Backward-compatible aliases from the first prototype API.
PUBLIC_DATA_SOURCES = {
    "nino34_csv": DRIVER_DATASETS["nino34"].url,
    "oisst_v2_monthly_1deg_nc": FIELD_DATASETS["oisst_v2_sst"].url,
    "coastline_zip": COASTLINE_URL,
}


def load_driver_nino34(csv_path: Path | str, config: SDCMapConfig) -> pd.Series:
    """Backward-compatible wrapper for the initial Niño3.4 loader."""
    return load_driver_series(csv_path, config=config, driver_key="nino34")


def load_sst_anomaly_subset(nc_path: Path | str, config: SDCMapConfig) -> xr.DataArray:
    """Backward-compatible wrapper for the initial OISST SST loader."""
    return load_field_anomaly_subset(nc_path, config=config, field_key="oisst_v2_sst")
