import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sdcpy_map.config import SDCMapConfig
from sdcpy_map.layers import (
    compute_sdcmap_event_layers,
    compute_sdcmap_layers,
    detect_driver_events,
)


def _synthetic_driver() -> pd.Series:
    index = pd.date_range("2000-01-01", periods=24, freq="MS")
    values = np.array(
        [
            0.0,
            0.2,
            1.4,
            0.4,
            0.0,
            -0.1,
            -0.2,
            -1.3,
            -0.2,
            0.0,
            0.1,
            1.1,
            0.2,
            0.0,
            -0.1,
            -1.0,
            -0.2,
            0.0,
            0.1,
            0.9,
            0.1,
            0.0,
            -0.1,
            -0.2,
        ],
        dtype=float,
    )
    return pd.Series(values, index=index)


def _window_bounds(event_idx: int, width: int) -> tuple[int, int]:
    half_before = (width - 1) // 2
    half_after = width - 1 - half_before
    return event_idx - half_before, event_idx + half_after + 1


def _synthetic_field() -> xr.DataArray:
    driver = _synthetic_driver()
    values = np.zeros((len(driver), 2, 2), dtype=float)

    selected_positive = [2, 11]
    selected_negative = [7, 15]
    ignored_positive = [19]
    width = 5

    # Cell (0, 0): only selected positive-event windows track the driver.
    for event_idx in selected_positive:
        start, stop = _window_bounds(event_idx, width)
        values[start:stop, 0, 0] = driver.iloc[start:stop].to_numpy(dtype=float)

    # Cell (1, 1): only selected negative-event windows track the driver.
    for event_idx in selected_negative:
        start, stop = _window_bounds(event_idx, width)
        values[start:stop, 1, 1] = driver.iloc[start:stop].to_numpy(dtype=float)

    # Cell (0, 1): only a non-selected ignored positive peak tracks the driver.
    for event_idx in ignored_positive:
        start, stop = _window_bounds(event_idx, width)
        values[start:stop, 0, 1] = driver.iloc[start:stop].to_numpy(dtype=float)
        if start - 2 >= 0:
            values[start - 2:start, 0, 1] = 7.0
        if stop + 2 <= len(driver):
            values[stop:stop + 2, 0, 1] = -7.0

    # Cell (1, 0): constant baseline, should stay invalid.
    values[:, 1, 0] = 0.0

    return xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={"time": driver.index, "lat": [0.0, 1.0], "lon": [10.0, 20.0]},
    )


def _offset_positive_field() -> xr.DataArray:
    driver = _synthetic_driver()
    values = np.zeros((len(driver), 1, 1), dtype=float)
    width = 5

    for event_idx in [2, 11]:
        start, stop = _window_bounds(event_idx + 1, width)
        values[start:stop, 0, 0] = driver.iloc[start:stop].to_numpy(dtype=float)
        if start - 1 >= 0:
            values[start - 1, 0, 0] = 6.0
        if stop < len(driver):
            values[stop, 0, 0] = -6.0

    return xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={"time": driver.index, "lat": [0.0], "lon": [10.0]},
    )


def test_detect_driver_events_separates_positive_and_negative_peaks():
    config = SDCMapConfig(correlation_width=5, n_positive_peaks=2, n_negative_peaks=2, base_state_beta=0.5)
    catalog = detect_driver_events(_synthetic_driver(), config)

    assert [item["date"] for item in catalog["selected_positive"]] == ["2000-03-01", "2000-12-01"]
    assert [item["date"] for item in catalog["selected_negative"]] == ["2000-08-01", "2001-04-01"]
    assert [item["date"] for item in catalog["ignored_positive"]] == ["2001-08-01"]
    assert catalog["base_state_threshold"] == pytest.approx(0.5)
    assert catalog["base_state_count"] > 0


def test_detect_driver_events_excludes_selected_event_windows_but_keeps_ignored_windows_available():
    config = SDCMapConfig(correlation_width=5, n_positive_peaks=2, n_negative_peaks=2, base_state_beta=0.5)
    catalog = detect_driver_events(_synthetic_driver(), config)

    mask = np.asarray(catalog["base_state_mask"], dtype=bool)
    # These months are low-amplitude and still inside selected event windows, so they must be excluded.
    assert not bool(mask[1])
    assert not bool(mask[10])
    # The ignored positive event is not part of the base-state exclusion window.
    assert bool(mask[18])
    # A quiet month outside any event window remains part of the base state.
    assert bool(mask[23])


def test_detect_driver_events_skips_boundary_extrema_without_full_window():
    index = pd.date_range("2000-01-01", periods=8, freq="MS")
    driver = pd.Series([0.0, 2.0, 0.2, 0.0, -0.1, -1.8, -0.2, 0.0], index=index, dtype=float)
    config = SDCMapConfig(correlation_width=5, n_positive_peaks=2, n_negative_peaks=2)

    catalog = detect_driver_events(driver, config)

    assert catalog["selected_positive"] == []
    assert [item["date"] for item in catalog["selected_negative"]] == ["2000-06-01"]
    assert any("Skipped" in warning for warning in catalog["warnings"])


def test_compute_sdcmap_event_layers_use_selected_event_windows_only(monkeypatch):
    monkeypatch.setattr(
        "sdcpy.compute_sdc",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("full-series compute_sdc should not be called")),
    )
    config = SDCMapConfig(
        correlation_width=5,
        n_positive_peaks=2,
        n_negative_peaks=2,
        base_state_beta=0.5,
        n_permutations=49,
        alpha=0.1,
        min_lag=0,
        max_lag=0,
    )

    result = compute_sdcmap_event_layers(
        driver=_synthetic_driver(),
        mapped_field=_synthetic_field(),
        config=config,
    )

    positive_corr = result["positive"]["layers"]["corr_mean"]
    negative_corr = result["negative"]["layers"]["corr_mean"]

    assert "positive" in result
    assert "negative" in result
    assert "event_catalog" in result
    assert result["positive"]["lag_maps"]["lags"] == [0]
    assert result["positive"]["lag_maps"]["corr_by_lag"].shape == (1, 2, 2)
    assert result["positive"]["summary"]["selected_event_count"] == 2
    assert result["negative"]["summary"]["selected_event_count"] == 2
    assert np.isfinite(positive_corr[0, 0])
    assert np.isnan(positive_corr[0, 1])
    assert np.isnan(positive_corr[1, 0])
    assert np.isfinite(negative_corr[1, 1])
    assert np.isnan(negative_corr[1, 0])
    assert result["positive"]["layers"]["lag_mean"][0, 0] == pytest.approx(0.0)
    assert result["negative"]["layers"]["lag_mean"][1, 1] == pytest.approx(0.0)
    assert result["positive"]["lag_maps"]["event_count_by_lag"][0, 0, 0] == pytest.approx(2.0)


def test_compute_sdcmap_layers_keeps_compact_compatibility(monkeypatch):
    monkeypatch.setattr(
        "sdcpy.compute_sdc",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("full-series compute_sdc should not be called")),
    )
    compact = compute_sdcmap_layers(
        driver=_synthetic_driver(),
        mapped_field=_synthetic_field(),
        config=SDCMapConfig(correlation_width=5, n_positive_peaks=2, n_negative_peaks=2, n_permutations=49, alpha=0.1),
    )

    assert compact["corr_mean"].shape == (2, 2)
    assert "dominant_sign" in compact
    assert np.isfinite(compact["corr_mean"][0, 0])


def test_compute_sdcmap_event_layers_tracks_peak_relative_position(monkeypatch):
    monkeypatch.setattr(
        "sdcpy.compute_sdc",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("full-series compute_sdc should not be called")),
    )
    result = compute_sdcmap_event_layers(
        driver=_synthetic_driver(),
        mapped_field=_offset_positive_field(),
        config=SDCMapConfig(
            correlation_width=5,
            n_positive_peaks=2,
            n_negative_peaks=0,
            base_state_beta=0.5,
            n_permutations=49,
            alpha=0.1,
            min_lag=-1,
            max_lag=1,
        ),
    )

    positive_layers = result["positive"]["layers"]
    assert positive_layers["driver_rel_time_mean"][0, 0] == pytest.approx(1.0)
    assert positive_layers["lag_mean"][0, 0] == pytest.approx(0.0)
    assert positive_layers["timing_combo"][0, 0] == pytest.approx(1.0)


def test_compute_sdcmap_event_layers_reports_cell_progress(monkeypatch):
    monkeypatch.setattr(
        "sdcpy.compute_sdc",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("full-series compute_sdc should not be called")),
    )
    updates: list[tuple[int, int]] = []
    compute_sdcmap_event_layers(
        driver=_synthetic_driver(),
        mapped_field=_synthetic_field(),
        config=SDCMapConfig(correlation_width=5, n_positive_peaks=2, n_negative_peaks=2, n_permutations=9, alpha=0.2),
        progress_callback=lambda current, total: updates.append((current, total)),
    )

    assert updates
    assert updates[-1] == (4, 4)


def test_compute_sdcmap_event_layers_requires_time_alignment():
    time = pd.date_range("2000-01-01", periods=12, freq="MS")
    driver = pd.Series(np.arange(10), index=time[:10])
    mapped = xr.DataArray(
        np.random.RandomState(0).randn(len(time), 1, 1),
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": [0.0], "lon": [0.0]},
    )

    with pytest.raises(ValueError):
        compute_sdcmap_event_layers(driver=driver, mapped_field=mapped, config=SDCMapConfig())
