from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString

from sdcpy_map.plotting import plot_layer_maps_compact


def _sample_inputs():
    lats = np.linspace(-20, 20, 6)
    lons = np.linspace(-170, -70, 8)
    shape = (len(lats), len(lons))
    rng = np.random.RandomState(7)
    layers = {
        "corr_mean": rng.uniform(-1, 1, size=shape),
        "lag_mean": rng.uniform(-6, 6, size=shape),
        "driver_rel_time_mean": rng.uniform(-10, 10, size=shape),
        "dominant_sign": rng.choice([-1.0, 1.0], size=shape),
    }
    coastline = gpd.GeoDataFrame(
        geometry=[LineString([(-170, -20), (-70, 20)])],
        crs="EPSG:4326",
    )
    return layers, lats, lons, coastline


def test_plot_layer_maps_compact_saves_png(tmp_path: Path):
    layers, lats, lons, coastline = _sample_inputs()
    out_path = tmp_path / "layers.png"

    ret = plot_layer_maps_compact(
        layers=layers,
        lats=lats,
        lons=lons,
        coastline=coastline,
        out_path=out_path,
    )

    assert ret == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_layer_maps_compact_returns_subplot_handles(tmp_path: Path):
    layers, lats, lons, coastline = _sample_inputs()
    out_path = tmp_path / "layers_handles.png"

    fig, axes, cbar_axes = plot_layer_maps_compact(
        layers=layers,
        lats=lats,
        lons=lons,
        coastline=coastline,
        out_path=out_path,
        return_handles=True,
    )

    assert axes.shape == (2, 2)
    assert len(cbar_axes) == 4
    assert axes[0, 0].get_xlabel() == ""
    assert axes[1, 0].get_xlabel() == "Longitude"
    assert axes[0, 1].get_ylabel() == ""
    assert axes[1, 1].get_ylabel() == ""

    fig.canvas.draw()
    fig.clf()
