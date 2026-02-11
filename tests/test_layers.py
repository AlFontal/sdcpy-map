import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sdcpy_map.config import SDCMapConfig
from sdcpy_map.layers import compute_sdcmap_layers


def _mock_sdc_df() -> pd.DataFrame:
    # Mix of positive/negative significant rows; top fraction determines selected subset.
    return pd.DataFrame(
        {
            "start_1": [5, 6, 7, 8, 5, 8],
            "stop_1": [17, 18, 19, 20, 17, 20],
            "start_2": [5, 6, 7, 8, 5, 8],
            "stop_2": [17, 18, 19, 20, 17, 20],
            "lag": [0, 1, 2, 3, 0, 3],
            "r": [0.90, 0.70, 0.40, 0.10, -0.60, -0.20],
            "p_value": [0.01, 0.01, 0.01, 0.01, 0.02, 0.02],
        }
    )


def test_compute_sdcmap_layers_uses_configured_tail(monkeypatch):
    captured: dict[str, object] = {}

    def fake_compute_sdc(*args, **kwargs):
        captured["two_tailed"] = kwargs["two_tailed"]
        return _mock_sdc_df()

    monkeypatch.setattr("sdcpy.compute_sdc", fake_compute_sdc)

    time = pd.date_range("2000-01-01", periods=12, freq="MS")
    driver = pd.Series(np.linspace(-1, 1, len(time)), index=time)
    mapped = xr.DataArray(
        np.random.RandomState(42).randn(len(time), 2, 2),
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": [0.0, 1.0], "lon": [10.0, 20.0]},
    )
    # Make one cell constant to force NaN summaries there.
    mapped[:, 0, 0] = 0.0

    config = SDCMapConfig(
        fragment_size=4,
        n_permutations=9,
        two_tailed=False,
        alpha=0.05,
        top_fraction=0.5,
        peak_date="2000-06-01",
    )
    layers = compute_sdcmap_layers(driver=driver, mapped_field=mapped, config=config)

    assert captured["two_tailed"] is False
    assert layers["corr_mean"].shape == (2, 2)
    assert np.isnan(layers["corr_mean"][0, 0])
    assert np.isfinite(layers["corr_mean"][1, 1])
    # Positive subset should dominate in this mocked scenario.
    assert layers["dominant_sign"][1, 1] == pytest.approx(1.0)


def test_compute_sdcmap_layers_requires_time_alignment(monkeypatch):
    monkeypatch.setattr("sdcpy.compute_sdc", lambda *args, **kwargs: _mock_sdc_df())

    time = pd.date_range("2000-01-01", periods=12, freq="MS")
    driver = pd.Series(np.arange(10), index=time[:10])
    mapped = xr.DataArray(
        np.random.RandomState(0).randn(len(time), 1, 1),
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": [0.0], "lon": [0.0]},
    )
    config = SDCMapConfig(fragment_size=4)

    with pytest.raises(ValueError):
        compute_sdcmap_layers(driver=driver, sst_anom=mapped, config=config)
