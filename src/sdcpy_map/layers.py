"""SDCMap-style layer computation."""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from sdcpy_map.config import SDCMapConfig


def _pick_top_extremes(
    sdc: pd.DataFrame,
    peak_idx: int,
    top_fraction: float,
) -> tuple[pd.DataFrame, str] | None:
    pos = sdc.loc[sdc["r"] > 0].copy()
    neg = sdc.loc[sdc["r"] < 0].copy()

    cpos = math.floor(len(pos) * top_fraction)
    cneg = math.floor(len(neg) * top_fraction)

    candidates: list[tuple[str, pd.DataFrame]] = []

    if cpos >= 2:
        sel_pos = pos.nlargest(cpos, "r").copy()
        sel_pos["driver_rel_start"] = sel_pos["start_1"] - peak_idx
        candidates.append(("pos", sel_pos))

    if cneg >= 2:
        sel_neg = neg.nsmallest(cneg, "r").copy()
        sel_neg["driver_rel_start"] = sel_neg["start_1"] - peak_idx
        candidates.append(("neg", sel_neg))

    if not candidates:
        return None

    if len(candidates) == 1:
        return candidates[0][1], candidates[0][0]

    sign, selected = max(candidates, key=lambda item: abs(item[1]["r"].mean()))
    return selected, sign


def _summarize_gridpoint(
    driver_vals: np.ndarray,
    local_vals: np.ndarray,
    config: SDCMapConfig,
    peak_idx: int,
) -> dict[str, float] | None:
    from sdcpy import compute_sdc

    if np.sum(np.isfinite(local_vals)) < config.fragment_size + 3:
        return None
    if np.nanstd(local_vals) == 0:
        return None
    if np.isnan(local_vals).any() or np.isnan(driver_vals).any():
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        sdc = compute_sdc(
            driver_vals,
            local_vals,
            fragment_size=config.fragment_size,
            n_permutations=config.n_permutations,
            two_tailed=config.two_tailed,
            min_lag=config.min_lag,
            max_lag=config.max_lag,
        )

    sdc = sdc.loc[np.isfinite(sdc["r"]) & np.isfinite(sdc["p_value"])]
    sdc = sdc.loc[sdc["p_value"] <= config.alpha]

    if sdc.empty:
        return None

    picked = _pick_top_extremes(sdc, peak_idx=peak_idx, top_fraction=config.top_fraction)
    if picked is None:
        return None

    selected, sign = picked

    rel = selected["driver_rel_start"].to_numpy(dtype=float)
    lag = selected["lag"].to_numpy(dtype=float)
    r = selected["r"].to_numpy(dtype=float)

    return {
        "corr_mean": float(np.mean(r)),
        "driver_rel_time_mean": float(np.mean(rel)),
        "lag_mean": float(np.mean(lag)),
        "timing_combo": float(np.mean(rel) + np.mean(lag)),
        "strong_span": float(np.max(rel) - np.min(rel)),
        "strong_start": float(np.min(rel)),
        "dominant_sign": 1.0 if sign == "pos" else -1.0,
        "n_selected": float(len(selected)),
    }


def compute_sdcmap_layers(
    driver: pd.Series,
    mapped_field: xr.DataArray | None = None,
    config: SDCMapConfig | None = None,
    *,
    sst_anom: xr.DataArray | None = None,
) -> dict[str, np.ndarray]:
    """Compute SDCMap-style summary layers for every gridpoint.

    Parameters
    ----------
    mapped_field
        Gridded mapped variable with dimensions ``(time, lat, lon)``.
    config
        SDCMap configuration parameters.
    sst_anom
        Backward-compatible alias for ``mapped_field``.
    """
    if mapped_field is None:
        mapped_field = sst_anom
    if mapped_field is None:
        raise ValueError("`mapped_field` must be provided.")
    if config is None:
        raise ValueError("`config` must be provided.")

    index = pd.DatetimeIndex(mapped_field["time"].values)
    driver = driver.reindex(index)
    if driver.isna().any():
        raise ValueError("Driver and mapped-variable time coverage do not align.")

    peak_date = pd.Timestamp(config.peak_date)
    peak_idx = int(np.argmin(np.abs(index - peak_date)))
    driver_vals = driver.to_numpy(dtype=float)

    nlat = mapped_field.sizes["lat"]
    nlon = mapped_field.sizes["lon"]

    layers = {
        "corr_mean": np.full((nlat, nlon), np.nan, dtype=float),
        "driver_rel_time_mean": np.full((nlat, nlon), np.nan, dtype=float),
        "lag_mean": np.full((nlat, nlon), np.nan, dtype=float),
        "timing_combo": np.full((nlat, nlon), np.nan, dtype=float),
        "strong_span": np.full((nlat, nlon), np.nan, dtype=float),
        "strong_start": np.full((nlat, nlon), np.nan, dtype=float),
        "dominant_sign": np.full((nlat, nlon), np.nan, dtype=float),
        "n_selected": np.full((nlat, nlon), np.nan, dtype=float),
    }

    for i in range(nlat):
        for j in range(nlon):
            local_vals = np.asarray(mapped_field[:, i, j].values, dtype=float)
            summary = _summarize_gridpoint(
                driver_vals=driver_vals,
                local_vals=local_vals,
                config=config,
                peak_idx=peak_idx,
            )
            if summary is None:
                continue
            for key, value in summary.items():
                layers[key][i, j] = value

    return layers


def save_layers_npz(
    path: Path | str,
    layers: dict[str, np.ndarray],
    lats: np.ndarray,
    lons: np.ndarray,
) -> Path:
    """Save layer outputs as compressed numpy archive."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, lat=lats, lon=lons, **layers)
    return path
