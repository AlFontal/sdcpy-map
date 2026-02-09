from pathlib import Path

import numpy as np

from sdcpy_map.layers import save_layers_npz


def test_save_layers_npz_roundtrip(tmp_path: Path):
    layers = {
        "corr_mean": np.zeros((2, 3), dtype=float),
        "driver_rel_time_mean": np.ones((2, 3), dtype=float),
    }
    lats = np.array([0.0, 1.0])
    lons = np.array([10.0, 20.0, 30.0])

    out = save_layers_npz(tmp_path / "layers.npz", layers=layers, lats=lats, lons=lons)
    loaded = np.load(out)

    assert loaded["lat"].shape == (2,)
    assert loaded["lon"].shape == (3,)
    assert loaded["corr_mean"].shape == (2, 3)
