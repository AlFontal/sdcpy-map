"""Event-based SDCMap layer computation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import xarray as xr

from sdcpy_map.config import SDCMapConfig

LAYER_KEYS = (
    "corr_mean",
    "driver_rel_time_mean",
    "lag_mean",
    "timing_combo",
    "strong_span",
    "strong_start",
    "dominant_sign",
    "n_selected",
)

LAG_MAP_KEYS = (
    "corr_by_lag",
    "event_count_by_lag",
)


def _empty_layers(nlat: int, nlon: int) -> dict[str, np.ndarray]:
    return {key: np.full((nlat, nlon), np.nan, dtype=float) for key in LAYER_KEYS}


def _serialize_event(event: dict[str, object]) -> dict[str, object]:
    return {
        "index": int(event["index"]),
        "date": str(event["date"]),
        "value": float(event["value"]),
        "sign": str(event["sign"]),
    }


def _iter_driver_extrema(driver: pd.Series) -> list[dict[str, object]]:
    values = driver.to_numpy(dtype=float)
    index = pd.DatetimeIndex(driver.index)
    extrema: list[dict[str, object]] = []
    if len(values) < 3:
        return extrema

    for idx in range(1, len(values) - 1):
        prev_value = values[idx - 1]
        current = values[idx]
        next_value = values[idx + 1]
        if not np.isfinite(current):
            continue
        if current > 0 and current >= prev_value and current > next_value:
            extrema.append(
                {
                    "index": idx,
                    "date": index[idx].date().isoformat(),
                    "value": float(current),
                    "sign": "positive",
                }
            )
        elif current < 0 and current <= prev_value and current < next_value:
            extrema.append(
                {
                    "index": idx,
                    "date": index[idx].date().isoformat(),
                    "value": float(current),
                    "sign": "negative",
                }
            )
    return extrema


def _event_window_bounds(event_idx: int, width: int, series_len: int) -> tuple[int, int] | None:
    half_before = (width - 1) // 2
    half_after = width - 1 - half_before
    start = int(event_idx) - half_before
    stop = int(event_idx) + half_after + 1
    if start < 0 or stop > int(series_len):
        return None
    return start, stop


def _iter_event_centers(event_idx: int, width: int, series_len: int) -> list[int]:
    """Return admissible fragment centers inside the local event neighborhood."""
    half_before = (int(width) - 1) // 2
    half_after = int(width) - 1 - half_before
    centers: list[int] = []
    for offset in range(-half_before, half_after + 1):
        center_idx = int(event_idx) + int(offset)
        if _event_window_bounds(center_idx, int(width), int(series_len)) is not None:
            centers.append(center_idx)
    return centers


def _select_event_subset(
    candidates: list[dict[str, object]],
    *,
    requested_count: int,
    min_separation: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    ranked = sorted(candidates, key=lambda item: abs(float(item["value"])), reverse=True)
    selected: list[dict[str, object]] = []
    ignored: list[dict[str, object]] = []
    for candidate in ranked:
        event_idx = int(candidate["index"])
        if any(abs(int(existing["index"]) - event_idx) < min_separation for existing in selected):
            ignored.append(candidate)
            continue
        if len(selected) < requested_count:
            selected.append(candidate)
        else:
            ignored.append(candidate)
    selected.sort(key=lambda item: int(item["index"]))
    ignored.sort(key=lambda item: abs(float(item["value"])), reverse=True)
    return selected, ignored


def detect_driver_events(driver: pd.Series, config: SDCMapConfig) -> dict[str, object]:
    """Detect positive and negative driver events plus a base-state mask."""
    clean_driver = driver.dropna().sort_index()
    raw_extrema = _iter_driver_extrema(clean_driver)
    full_len = len(clean_driver)
    extrema = [
        item
        for item in raw_extrema
        if _event_window_bounds(int(item["index"]), int(config.correlation_width), full_len) is not None
    ]
    positive_candidates = [item for item in extrema if item["sign"] == "positive"]
    negative_candidates = [item for item in extrema if item["sign"] == "negative"]

    min_separation = max(1, int(config.correlation_width))
    selected_positive, ignored_positive = _select_event_subset(
        positive_candidates,
        requested_count=int(config.n_positive_peaks),
        min_separation=min_separation,
    )
    selected_negative, ignored_negative = _select_event_subset(
        negative_candidates,
        requested_count=int(config.n_negative_peaks),
        min_separation=min_separation,
    )

    selected_all = selected_positive + selected_negative
    warnings_out: list[str] = []
    dropped_for_window = len(raw_extrema) - len(extrema)
    if dropped_for_window:
        warnings_out.append(
            f"Skipped {dropped_for_window} extrema that could not support a full event window of width {int(config.correlation_width)}."
        )
    if len(selected_positive) < int(config.n_positive_peaks):
        warnings_out.append(
            f"Requested {int(config.n_positive_peaks)} positive events, but found {len(selected_positive)} usable extrema."
        )
    if len(selected_negative) < int(config.n_negative_peaks):
        warnings_out.append(
            f"Requested {int(config.n_negative_peaks)} negative events, but found {len(selected_negative)} usable extrema."
        )

    threshold: float | None = None
    if selected_all:
        threshold = float(
            config.base_state_beta
            * min(abs(float(item["value"])) for item in selected_all)
        )

    full_driver = driver.sort_index()
    driver_values = full_driver.to_numpy(dtype=float)
    if threshold is None or not np.isfinite(threshold):
        base_state_mask = np.isfinite(driver_values)
    else:
        base_state_mask = np.isfinite(driver_values) & (np.abs(driver_values) < threshold)

    exclusion_events = selected_positive + selected_negative
    for event in exclusion_events:
        bounds = _event_window_bounds(int(event["index"]), int(config.correlation_width), len(driver_values))
        if bounds is None:
            continue
        start, stop = bounds
        base_state_mask[start:stop] = False
    base_state_count = int(np.sum(base_state_mask))
    if base_state_count == 0:
        base_state_mask = np.isfinite(driver_values)
        base_state_count = int(np.sum(base_state_mask))
        warnings_out.append(
            "No baseline samples remained after excluding selected event windows; using all aligned time steps as fallback."
        )

    return {
        "selected_positive": [_serialize_event(item) for item in selected_positive],
        "selected_negative": [_serialize_event(item) for item in selected_negative],
        "ignored_positive": [_serialize_event(item) for item in ignored_positive],
        "ignored_negative": [_serialize_event(item) for item in ignored_negative],
        "base_state_threshold": threshold,
        "base_state_count": base_state_count,
        "base_state_mask": base_state_mask,
        "selected_positive_indices": [int(item["index"]) for item in selected_positive],
        "selected_negative_indices": [int(item["index"]) for item in selected_negative],
        "warnings": warnings_out,
    }


def _center_rows(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = values - np.mean(values, axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    return centered, norms


def _correlate_reference_against_rows(reference: np.ndarray, rows: np.ndarray) -> np.ndarray:
    reference_matrix = np.asarray(reference, dtype=float).reshape(1, -1)
    row_matrix = np.asarray(rows, dtype=float)
    ref_centered, ref_norms = _center_rows(reference_matrix)
    row_centered, row_norms = _center_rows(row_matrix)
    denom = ref_norms[0] * row_norms
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = (row_centered @ ref_centered[0]) / denom
    corr[~np.isfinite(denom) | (denom == 0)] = np.nan
    return corr


def _correlation_matrix(left_rows: np.ndarray, right_rows: np.ndarray) -> np.ndarray:
    left_centered, left_norms = _center_rows(np.asarray(left_rows, dtype=float))
    right_centered, right_norms = _center_rows(np.asarray(right_rows, dtype=float))
    denom = np.outer(left_norms, right_norms)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = (left_centered @ right_centered.T) / denom
    corr[~np.isfinite(denom) | (denom == 0)] = np.nan
    return corr


def _permutation_p_values(
    driver_segment: np.ndarray,
    field_segments: np.ndarray,
    observed: np.ndarray,
    *,
    n_permutations: int,
    two_tailed: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    if n_permutations < 1:
        return np.full_like(observed, np.nan, dtype=float)

    permutations = np.vstack([rng.permutation(driver_segment) for _ in range(n_permutations)])
    perm_corr = _correlation_matrix(permutations, field_segments)

    observed_vals = np.asarray(observed, dtype=float)
    counts = np.zeros_like(observed_vals, dtype=float)
    finite_mask = np.isfinite(observed_vals)
    if two_tailed:
        counts[finite_mask] = np.sum(
            np.abs(perm_corr[:, finite_mask]) >= np.abs(observed_vals[finite_mask])[None, :],
            axis=0,
        )
    else:
        positive_mask = finite_mask & (observed_vals >= 0)
        negative_mask = finite_mask & (observed_vals < 0)
        if np.any(positive_mask):
            counts[positive_mask] = np.sum(
                perm_corr[:, positive_mask] >= observed_vals[positive_mask][None, :],
                axis=0,
            )
        if np.any(negative_mask):
            counts[negative_mask] = np.sum(
                perm_corr[:, negative_mask] <= observed_vals[negative_mask][None, :],
                axis=0,
            )
    p_values = (counts + 1.0) / float(n_permutations + 1)
    p_values[~finite_mask] = np.nan
    return p_values


def _candidate_score(
    corr_value: float,
    lag_value: float,
    driver_rel_time: float,
) -> tuple[float, float, float, float, float]:
    """Order candidates by strength, then by simpler timing."""
    return (
        -abs(float(corr_value)),
        abs(float(lag_value)),
        abs(float(driver_rel_time)),
        float(lag_value),
        float(driver_rel_time),
    )


def _aggregate_event_summaries(items: list[dict[str, float]]) -> dict[str, float] | None:
    if not items:
        return None
    keys = items[0].keys()
    out = {key: float(np.mean([item[key] for item in items])) for key in keys}
    out["n_selected"] = float(len(items))
    out["dominant_sign"] = 1.0 if out["corr_mean"] >= 0 else -1.0
    return out


def _summarize_event_window(
    driver_vals: np.ndarray,
    local_vals: np.ndarray,
    *,
    event_idx: int,
    config: SDCMapConfig,
    rng: np.random.Generator,
) -> dict[str, float] | None:
    best_summary: dict[str, float] | None = None
    best_score: tuple[float, float, float, float, float] | None = None

    for center_idx in _iter_event_centers(int(event_idx), int(config.correlation_width), len(driver_vals)):
        bounds = _event_window_bounds(center_idx, int(config.correlation_width), len(driver_vals))
        if bounds is None:
            continue
        start_1, stop_1 = bounds
        driver_segment = np.asarray(driver_vals[start_1:stop_1], dtype=float)
        if np.isnan(driver_segment).any() or np.nanstd(driver_segment) == 0:
            continue

        lag_values: list[int] = []
        field_segments: list[np.ndarray] = []
        for lag in range(int(config.min_lag), int(config.max_lag) + 1):
            start_2 = start_1 - int(lag)
            stop_2 = start_2 + int(config.correlation_width)
            if start_2 < 0 or stop_2 > len(local_vals):
                continue
            field_segment = np.asarray(local_vals[start_2:stop_2], dtype=float)
            if np.isnan(field_segment).any() or np.nanstd(field_segment) == 0:
                continue
            lag_values.append(int(lag))
            field_segments.append(field_segment)
        if not field_segments:
            continue

        field_matrix = np.asarray(field_segments, dtype=float)
        observed = _correlate_reference_against_rows(driver_segment, field_matrix)
        p_values = _permutation_p_values(
            driver_segment,
            field_matrix,
            observed,
            n_permutations=int(config.n_permutations),
            two_tailed=bool(config.two_tailed),
            rng=rng,
        )
        significant = np.isfinite(observed) & np.isfinite(p_values) & (p_values <= float(config.alpha))
        if not np.any(significant):
            continue

        driver_rel_center = float(int(center_idx) - int(event_idx))
        for idx in np.flatnonzero(significant):
            corr_value = float(observed[idx])
            lag_value = float(lag_values[idx])
            score = _candidate_score(corr_value, lag_value, driver_rel_center)
            if best_score is not None and score >= best_score:
                continue
            best_score = score
            best_summary = {
                "corr_mean": corr_value,
                "driver_rel_time_mean": driver_rel_center,
                "lag_mean": lag_value,
                "timing_combo": driver_rel_center - lag_value,
                "strong_span": float(int(config.correlation_width) - 1),
                "strong_start": float(start_1 - int(event_idx)),
                "dominant_sign": 1.0 if corr_value >= 0 else -1.0,
                "n_selected": 1.0,
            }

    return best_summary


def _compute_event_lag_correlations(
    driver_vals: np.ndarray,
    local_vals: np.ndarray,
    *,
    event_idx: int,
    config: SDCMapConfig,
    lag_values: list[int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Compute significant event-local correlations for every requested lag."""
    out = np.full(len(lag_values), np.nan, dtype=float)
    best_scores: list[tuple[float, float, float, float, float] | None] = [None] * len(lag_values)

    for center_idx in _iter_event_centers(int(event_idx), int(config.correlation_width), len(driver_vals)):
        bounds = _event_window_bounds(center_idx, int(config.correlation_width), len(driver_vals))
        if bounds is None:
            continue
        start_1, stop_1 = bounds
        driver_segment = np.asarray(driver_vals[start_1:stop_1], dtype=float)
        if np.isnan(driver_segment).any() or np.nanstd(driver_segment) == 0:
            continue

        field_segments: list[np.ndarray] = []
        valid_positions: list[int] = []
        for idx, lag in enumerate(lag_values):
            start_2 = start_1 - int(lag)
            stop_2 = start_2 + int(config.correlation_width)
            if start_2 < 0 or stop_2 > len(local_vals):
                continue
            field_segment = np.asarray(local_vals[start_2:stop_2], dtype=float)
            if np.isnan(field_segment).any() or np.nanstd(field_segment) == 0:
                continue
            valid_positions.append(idx)
            field_segments.append(field_segment)

        if not field_segments:
            continue

        field_matrix = np.asarray(field_segments, dtype=float)
        observed = _correlate_reference_against_rows(driver_segment, field_matrix)
        p_values = _permutation_p_values(
            driver_segment,
            field_matrix,
            observed,
            n_permutations=int(config.n_permutations),
            two_tailed=bool(config.two_tailed),
            rng=rng,
        )
        significant = np.isfinite(observed) & np.isfinite(p_values) & (p_values <= float(config.alpha))
        if not np.any(significant):
            continue

        driver_rel_center = float(int(center_idx) - int(event_idx))
        for local_idx, is_significant in enumerate(significant):
            if not is_significant:
                continue
            lag_position = int(valid_positions[local_idx])
            corr_value = float(observed[local_idx])
            lag_value = float(lag_values[lag_position])
            score = _candidate_score(corr_value, lag_value, driver_rel_center)
            if best_scores[lag_position] is not None and score >= best_scores[lag_position]:
                continue
            best_scores[lag_position] = score
            out[lag_position] = corr_value
    return out


def _summarize_gridpoint_by_class(
    driver_vals: np.ndarray,
    local_vals: np.ndarray,
    config: SDCMapConfig,
    event_catalog: dict[str, object],
    *,
    rng: np.random.Generator,
) -> dict[str, dict[str, float] | None]:
    if np.sum(np.isfinite(local_vals)) < config.correlation_width + 3:
        return {"positive": None, "negative": None}
    if np.nanstd(local_vals) == 0:
        return {"positive": None, "negative": None}
    if np.isnan(local_vals).any() or np.isnan(driver_vals).any():
        return {"positive": None, "negative": None}

    out: dict[str, dict[str, float] | None] = {}
    for sign_key, indices_key in (
        ("positive", "selected_positive_indices"),
        ("negative", "selected_negative_indices"),
    ):
        event_summaries: list[dict[str, float]] = []
        for event_idx in event_catalog[indices_key]:
            summary = _summarize_event_window(
                driver_vals,
                local_vals,
                event_idx=int(event_idx),
                config=config,
                rng=rng,
            )
            if summary is not None:
                event_summaries.append(summary)
        out[sign_key] = _aggregate_event_summaries(event_summaries)
    return out


def _build_compact_layers_from_lag_stack(
    corr_by_lag: np.ndarray,
    event_count_by_lag: np.ndarray,
    lag_values: np.ndarray,
    correlation_width: int,
) -> dict[str, np.ndarray]:
    """Collapse lag-resolved class maps into legacy per-cell summary layers."""
    nlag, nlat, nlon = corr_by_lag.shape
    compact = _empty_layers(nlat, nlon)
    if nlag == 0:
        return compact

    abs_corr = np.abs(corr_by_lag)
    finite_mask = np.isfinite(abs_corr)
    lag_priority = np.abs(lag_values)[:, None, None] + (lag_values[:, None, None] * 1e-6)
    score = np.where(finite_mask, abs_corr * 1_000_000.0 - lag_priority, -np.inf)
    best_indices = np.argmax(score, axis=0)
    has_any = np.any(finite_mask, axis=0)

    row_idx = np.arange(nlat)[:, None]
    col_idx = np.arange(nlon)[None, :]
    best_corr = corr_by_lag[best_indices, row_idx, col_idx]
    best_lag = lag_values[best_indices]
    best_count = event_count_by_lag[best_indices, row_idx, col_idx]

    compact["corr_mean"] = np.where(has_any, best_corr, np.nan)
    compact["lag_mean"] = np.where(has_any, best_lag, np.nan)
    compact["driver_rel_time_mean"] = np.where(has_any, 0.0, np.nan)
    compact["timing_combo"] = np.where(has_any, -best_lag, np.nan)
    compact["strong_span"] = np.where(has_any, float(int(correlation_width) - 1), np.nan)
    compact["strong_start"] = np.where(has_any, -float((int(correlation_width) - 1) // 2), np.nan)
    compact["dominant_sign"] = np.where(
        has_any,
        np.where(best_corr >= 0.0, 1.0, -1.0),
        np.nan,
    )
    compact["n_selected"] = np.where(has_any, best_count, np.nan)
    return compact


def _apply_base_state_filter(
    mapped_field: xr.DataArray,
    event_catalog: dict[str, object],
) -> tuple[xr.DataArray, dict[str, object]]:
    base_state_mask = np.asarray(event_catalog["base_state_mask"], dtype=bool)
    if base_state_mask.size != int(mapped_field.sizes["time"]):
        raise ValueError("Driver event catalog is not aligned to mapped-field time coverage.")
    if not np.any(base_state_mask):
        return mapped_field, event_catalog
    baseline = mapped_field.isel(time=np.where(base_state_mask)[0]).mean(dim="time")
    return mapped_field - baseline, event_catalog


def compute_sdcmap_event_layers(
    driver: pd.Series,
    mapped_field: xr.DataArray | None = None,
    config: SDCMapConfig | None = None,
    *,
    sst_anom: xr.DataArray | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    """Compute event-conditioned SDCMap layers for positive and negative driver classes."""
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

    event_catalog = detect_driver_events(driver, config)
    filtered_field, event_catalog = _apply_base_state_filter(mapped_field, event_catalog)
    driver_vals = driver.to_numpy(dtype=float)
    rng = np.random.default_rng(0)
    lag_values = np.arange(int(config.min_lag), int(config.max_lag) + 1, dtype=int)

    nlat = int(filtered_field.sizes["lat"])
    nlon = int(filtered_field.sizes["lon"])
    total_cells = max(0, nlat * nlon)
    callback_every = max(1, total_cells // 200) if total_cells else 1
    class_layers = {
        "positive": _empty_layers(nlat, nlon),
        "negative": _empty_layers(nlat, nlon),
    }
    class_lag_maps = {
        "positive": {
            "corr_by_lag": np.full((len(lag_values), nlat, nlon), np.nan, dtype=float),
            "event_count_by_lag": np.zeros((len(lag_values), nlat, nlon), dtype=float),
        },
        "negative": {
            "corr_by_lag": np.full((len(lag_values), nlat, nlon), np.nan, dtype=float),
            "event_count_by_lag": np.zeros((len(lag_values), nlat, nlon), dtype=float),
        },
    }

    completed_cells = 0
    for i in range(nlat):
        for j in range(nlon):
            local_vals = np.asarray(filtered_field[:, i, j].values, dtype=float)
            if (
                np.sum(np.isfinite(local_vals)) < config.correlation_width + 3
                or np.nanstd(local_vals) == 0
                or np.isnan(local_vals).any()
                or np.isnan(driver_vals).any()
            ):
                completed_cells += 1
                if progress_callback and (
                    completed_cells == total_cells or completed_cells % callback_every == 0
                ):
                    progress_callback(completed_cells, total_cells)
                continue

            summary_by_class = _summarize_gridpoint_by_class(
                driver_vals,
                local_vals,
                config,
                event_catalog,
                rng=rng,
            )
            for sign_key, indices_key in (
                ("positive", "selected_positive_indices"),
                ("negative", "selected_negative_indices"),
            ):
                class_summary = summary_by_class.get(sign_key)
                if class_summary is not None:
                    for key, value in class_summary.items():
                        if key in class_layers[sign_key]:
                            class_layers[sign_key][key][i, j] = float(value)
                event_corr_rows: list[np.ndarray] = []
                for event_idx in event_catalog[indices_key]:
                    event_corr_rows.append(
                        _compute_event_lag_correlations(
                            driver_vals,
                            local_vals,
                            event_idx=int(event_idx),
                            config=config,
                            lag_values=lag_values.tolist(),
                            rng=rng,
                        )
                    )
                if not event_corr_rows:
                    continue
                event_corr_matrix = np.asarray(event_corr_rows, dtype=float)
                finite_mask = np.isfinite(event_corr_matrix)
                if not np.any(finite_mask):
                    continue
                counts = np.sum(finite_mask, axis=0).astype(float)
                sums = np.where(finite_mask, event_corr_matrix, 0.0).sum(axis=0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    mean_corr = sums / counts
                mean_corr = np.where(counts > 0, mean_corr, np.nan)
                class_lag_maps[sign_key]["corr_by_lag"][:, i, j] = mean_corr
                class_lag_maps[sign_key]["event_count_by_lag"][:, i, j] = counts
            completed_cells += 1
            if progress_callback and (
                completed_cells == total_cells or completed_cells % callback_every == 0
            ):
                progress_callback(completed_cells, total_cells)

    def _class_summary(sign_key: str) -> dict[str, object]:
        corr = np.asarray(class_layers[sign_key]["corr_mean"], dtype=float)
        valid_cells = int(np.isfinite(corr).sum())
        requested = int(
            config.n_positive_peaks if sign_key == "positive" else config.n_negative_peaks
        )
        selected = len(event_catalog[f"selected_{sign_key}"])
        corr_by_lag = np.asarray(class_lag_maps[sign_key]["corr_by_lag"], dtype=float)
        lag_valid_cells = [
            int(np.isfinite(corr_by_lag[idx]).sum()) for idx in range(len(lag_values))
        ]
        return {
            "requested_event_count": requested,
            "selected_event_count": selected,
            "valid_cells": valid_cells,
            "valid_cell_rate": float(valid_cells / corr.size) if corr.size else 0.0,
            "mean_abs_corr": float(np.nanmean(np.abs(corr))) if np.isfinite(corr).any() else None,
            "lag_valid_cells": lag_valid_cells,
        }

    public_catalog = {
        "selected_positive": event_catalog["selected_positive"],
        "selected_negative": event_catalog["selected_negative"],
        "ignored_positive": event_catalog["ignored_positive"],
        "ignored_negative": event_catalog["ignored_negative"],
        "base_state_threshold": event_catalog["base_state_threshold"],
        "base_state_count": event_catalog["base_state_count"],
        "warnings": event_catalog["warnings"],
    }

    return {
        "positive": {
            "layers": class_layers["positive"],
            "lag_maps": {
                "lags": [int(item) for item in lag_values.tolist()],
                **{
                    key: np.asarray(values, dtype=float)
                    for key, values in class_lag_maps["positive"].items()
                },
            },
            "summary": _class_summary("positive"),
            "events": event_catalog["selected_positive"],
        },
        "negative": {
            "layers": class_layers["negative"],
            "lag_maps": {
                "lags": [int(item) for item in lag_values.tolist()],
                **{
                    key: np.asarray(values, dtype=float)
                    for key, values in class_lag_maps["negative"].items()
                },
            },
            "summary": _class_summary("negative"),
            "events": event_catalog["selected_negative"],
        },
        "event_catalog": public_catalog,
    }


def derive_compact_layers(event_result: dict[str, object]) -> dict[str, np.ndarray]:
    """Derive the deprecated compact 4-layer-style output from event-class results."""
    positive_layers = event_result["positive"]["layers"]
    negative_layers = event_result["negative"]["layers"]
    corr_pos = np.asarray(positive_layers["corr_mean"], dtype=float)
    corr_neg = np.asarray(negative_layers["corr_mean"], dtype=float)
    choose_positive = np.abs(corr_pos) >= np.abs(corr_neg)
    choose_positive = np.where(np.isnan(corr_neg), True, choose_positive)
    choose_positive = np.where(np.isnan(corr_pos), False, choose_positive)

    compact = _empty_layers(*corr_pos.shape)
    for key in LAYER_KEYS:
        pos_vals = np.asarray(positive_layers[key], dtype=float)
        neg_vals = np.asarray(negative_layers[key], dtype=float)
        compact[key] = np.where(choose_positive, pos_vals, neg_vals)
    compact["dominant_sign"] = np.where(
        np.isnan(compact["corr_mean"]),
        np.nan,
        np.where(choose_positive, 1.0, -1.0),
    )
    return compact


def compute_sdcmap_layers(
    driver: pd.Series,
    mapped_field: xr.DataArray | None = None,
    config: SDCMapConfig | None = None,
    *,
    sst_anom: xr.DataArray | None = None,
) -> dict[str, np.ndarray]:
    """Deprecated compatibility wrapper returning the compact combined output."""
    event_result = compute_sdcmap_event_layers(
        driver=driver,
        mapped_field=mapped_field,
        config=config,
        sst_anom=sst_anom,
    )
    return derive_compact_layers(event_result)


def save_layers_npz(
    path: Path | str,
    layers: dict[str, object],
    lats: np.ndarray,
    lons: np.ndarray,
) -> Path:
    """Save compact or event-class layer outputs as compressed numpy archive."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, object] = {
        "lat": np.asarray(lats, dtype=float),
        "lon": np.asarray(lons, dtype=float),
    }

    if "positive" in layers and "negative" in layers:
        payload["event_catalog_json"] = np.asarray(
            [json.dumps(layers.get("event_catalog", {}))],
            dtype=object,
        )
        for sign_key in ("positive", "negative"):
            sign_layers = layers[sign_key]["layers"]
            for key, values in sign_layers.items():
                payload[f"{sign_key}__{key}"] = np.asarray(values, dtype=float)
            lag_maps = layers[sign_key].get("lag_maps") or {}
            if lag_maps:
                payload[f"{sign_key}__lags"] = np.asarray(lag_maps.get("lags") or [], dtype=int)
                for key in LAG_MAP_KEYS:
                    if key in lag_maps:
                        payload[f"{sign_key}__{key}"] = np.asarray(lag_maps[key], dtype=float)
    else:
        for key, values in layers.items():
            payload[key] = np.asarray(values, dtype=float)

    np.savez_compressed(path, **payload)
    return path
