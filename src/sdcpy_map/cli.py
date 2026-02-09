"""Command-line demo for sdcpy-map."""

from __future__ import annotations

import argparse
from pathlib import Path

from sdcpy_map.config import SDCMapConfig
from sdcpy_map.datasets import (
    align_driver_to_field,
    fetch_public_example_data,
    grid_coordinates,
    load_coastline,
    load_driver_nino34,
    load_sst_anomaly_subset,
)
from sdcpy_map.layers import compute_sdcmap_layers, save_layers_npz
from sdcpy_map.plotting import plot_layer_maps_compact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a public-data SDCMap demo")
    parser.add_argument("--data-dir", type=Path, default=Path(".data"))
    parser.add_argument("--out-dir", type=Path, default=Path(".output"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SDCMapConfig()

    paths = fetch_public_example_data(args.data_dir)

    driver = load_driver_nino34(paths["nino34_csv"], config)
    sst_anom = load_sst_anomaly_subset(paths["ersstv5_nc"], config)
    driver = align_driver_to_field(driver, sst_anom)

    layers = compute_sdcmap_layers(driver=driver, sst_anom=sst_anom, config=config)

    lats, lons = grid_coordinates(sst_anom)

    npz_path = save_layers_npz(args.out_dir / "sdcmap_layers.npz", layers=layers, lats=lats, lons=lons)

    coastline = load_coastline(paths["coastline_zip"])
    png_path = plot_layer_maps_compact(
        layers=layers,
        lats=lats,
        lons=lons,
        coastline=coastline,
        out_path=args.out_dir / "sdcmap_layers.png",
    )

    print(f"Saved: {npz_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
