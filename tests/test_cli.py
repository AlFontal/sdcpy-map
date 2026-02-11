from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from sdcpy_map import cli


def test_parse_args_defaults():
    args = cli.parse_args([])
    assert args.driver_dataset == "pdo"
    assert args.field_dataset == "ncep_air"


def test_main_wires_selected_datasets(monkeypatch, tmp_path: Path, capsys):
    args = Namespace(
        data_dir=tmp_path / "data",
        out_dir=tmp_path / "out",
        driver_dataset="nao",
        field_dataset="ncep_air",
    )
    monkeypatch.setattr(cli, "parse_args", lambda argv=None: args)

    called: dict[str, object] = {}

    def fake_fetch_public_example_data(data_dir, driver_key, field_key, include_coastline=True):
        called["fetch"] = (driver_key, field_key)
        return {"driver": Path("driver.data"), "field": Path("field.nc"), "coastline": Path("coast.zip")}

    def fake_load_driver_series(path, config, driver_key):
        called["driver_key"] = driver_key
        idx = pd.date_range(config.time_start, periods=24, freq="MS")
        return pd.Series(np.linspace(-1, 1, len(idx)), index=idx)

    def fake_load_field_anomaly_subset(path, config, field_key):
        called["field_key"] = field_key
        idx = pd.date_range(config.time_start, periods=24, freq="MS")
        return xr.DataArray(
            np.random.RandomState(0).randn(len(idx), 2, 2),
            dims=("time", "lat", "lon"),
            coords={"time": idx, "lat": [0.0, 1.0], "lon": [10.0, 20.0]},
        )

    def fake_align(driver, mapped):
        return driver

    def fake_compute(*, driver, mapped_field, config):
        return {
            "corr_mean": np.zeros((2, 2), dtype=float),
            "lag_mean": np.zeros((2, 2), dtype=float),
            "driver_rel_time_mean": np.zeros((2, 2), dtype=float),
            "dominant_sign": np.ones((2, 2), dtype=float),
        }

    def fake_grid(mapped):
        return np.array([0.0, 1.0]), np.array([10.0, 20.0])

    def fake_save(path, layers, lats, lons):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("npz", encoding="utf-8")
        return path

    def fake_load_coast(path):
        return None

    def fake_plot(*, layers, lats, lons, coastline, out_path, title=""):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("png", encoding="utf-8")
        return out_path

    monkeypatch.setattr(cli, "fetch_public_example_data", fake_fetch_public_example_data)
    monkeypatch.setattr(cli, "load_driver_series", fake_load_driver_series)
    monkeypatch.setattr(cli, "load_field_anomaly_subset", fake_load_field_anomaly_subset)
    monkeypatch.setattr(cli, "align_driver_to_field", fake_align)
    monkeypatch.setattr(cli, "compute_sdcmap_layers", fake_compute)
    monkeypatch.setattr(cli, "grid_coordinates", fake_grid)
    monkeypatch.setattr(cli, "save_layers_npz", fake_save)
    monkeypatch.setattr(cli, "load_coastline", fake_load_coast)
    monkeypatch.setattr(cli, "plot_layer_maps_compact", fake_plot)

    cli.main()
    out = capsys.readouterr().out

    assert called["fetch"] == ("nao", "ncep_air")
    assert called["driver_key"] == "nao"
    assert called["field_key"] == "ncep_air"
    assert "Driver dataset: nao" in out
    assert "Field dataset: ncep_air" in out
