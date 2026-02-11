"""Top-level package for sdcpy-map."""

__author__ = "Alejandro Fontal"
__version__ = "0.1.0"

from sdcpy_map.config import SDCMapConfig
from sdcpy_map.datasets import (
    COASTLINE_URL,
    DEFAULT_DRIVER_DATASET_KEY,
    DEFAULT_FIELD_DATASET_KEY,
    DRIVER_DATASETS,
    FIELD_DATASETS,
    PUBLIC_DATA_SOURCES,
    align_driver_to_field,
    fetch_public_example_data,
    grid_coordinates,
    load_coastline,
    load_driver_nino34,
    load_driver_series,
    load_field_anomaly_subset,
    load_sst_anomaly_subset,
)
from sdcpy_map.layers import compute_sdcmap_layers, save_layers_npz
from sdcpy_map.plotting import plot_layer_maps_compact

__all__ = [
    "SDCMapConfig",
    "DRIVER_DATASETS",
    "FIELD_DATASETS",
    "DEFAULT_DRIVER_DATASET_KEY",
    "DEFAULT_FIELD_DATASET_KEY",
    "COASTLINE_URL",
    "PUBLIC_DATA_SOURCES",
    "fetch_public_example_data",
    "load_driver_series",
    "load_field_anomaly_subset",
    "load_driver_nino34",
    "load_sst_anomaly_subset",
    "load_coastline",
    "align_driver_to_field",
    "grid_coordinates",
    "compute_sdcmap_layers",
    "save_layers_npz",
    "plot_layer_maps_compact",
]
