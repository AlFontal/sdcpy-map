"""sdcpy-map: spatial SDC map utilities."""

from sdcpy_map.config import SDCMapConfig
from sdcpy_map.datasets import (
    PUBLIC_DATA_SOURCES,
    fetch_public_example_data,
    load_coastline,
    load_driver_nino34,
    load_sst_anomaly_subset,
)
from sdcpy_map.layers import compute_sdcmap_layers, save_layers_npz
from sdcpy_map.plotting import plot_layer_maps_compact

__all__ = [
    "SDCMapConfig",
    "PUBLIC_DATA_SOURCES",
    "fetch_public_example_data",
    "load_driver_nino34",
    "load_sst_anomaly_subset",
    "load_coastline",
    "compute_sdcmap_layers",
    "save_layers_npz",
    "plot_layer_maps_compact",
]
