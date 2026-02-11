"""Command-line demo for sdcpy-map."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from sdcpy_map.config import SDCMapConfig
from sdcpy_map.datasets import (
    DEFAULT_DRIVER_DATASET_KEY,
    DEFAULT_FIELD_DATASET_KEY,
    DRIVER_DATASETS,
    FIELD_DATASETS,
    align_driver_to_field,
    fetch_public_example_data,
    grid_coordinates,
    load_coastline,
    load_driver_series,
    load_field_anomaly_subset,
)
from sdcpy_map.layers import compute_sdcmap_layers, save_layers_npz
from sdcpy_map.plotting import plot_layer_maps_compact


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a public-data SDCMap demo")
    parser.add_argument("--data-dir", type=Path, default=Path(".data"))
    parser.add_argument("--out-dir", type=Path, default=Path(".output"))
    parser.add_argument(
        "--driver-dataset",
        type=str,
        default=DEFAULT_DRIVER_DATASET_KEY,
        choices=sorted(DRIVER_DATASETS),
        help="Driver index source key.",
    )
    parser.add_argument(
        "--field-dataset",
        type=str,
        default=DEFAULT_FIELD_DATASET_KEY,
        choices=sorted(FIELD_DATASETS),
        help="Mapped-variable source key.",
    )
    return parser.parse_args(argv)


def main() -> None:
    # Suppress per-gridpoint tqdm bars from sdcpy internals in CLI mode.
    os.environ.setdefault("TQDM_DISABLE", "1")

    args = parse_args()
    config = SDCMapConfig()

    paths = fetch_public_example_data(
        args.data_dir,
        driver_key=args.driver_dataset,
        field_key=args.field_dataset,
    )

    driver = load_driver_series(paths["driver"], config=config, driver_key=args.driver_dataset)
    mapped_field = load_field_anomaly_subset(paths["field"], config=config, field_key=args.field_dataset)
    driver = align_driver_to_field(driver, mapped_field)

    layers = compute_sdcmap_layers(driver=driver, mapped_field=mapped_field, config=config)

    lats, lons = grid_coordinates(mapped_field)

    npz_path = save_layers_npz(args.out_dir / "sdcmap_layers.npz", layers=layers, lats=lats, lons=lons)

    coastline = load_coastline(paths["coastline"])
    png_path = plot_layer_maps_compact(
        layers=layers,
        lats=lats,
        lons=lons,
        coastline=coastline,
        out_path=args.out_dir / "sdcmap_layers.png",
    )

    print(f"Driver dataset: {args.driver_dataset}")
    print(f"Field dataset: {args.field_dataset}")
    print(f"Saved: {npz_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
