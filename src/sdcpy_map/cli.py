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
from sdcpy_map.layers import compute_sdcmap_event_layers, save_layers_npz
from sdcpy_map.plotting import (
    plot_correlation_maps_by_lag,
    plot_layer_maps_compact,
    plot_single_layer_map,
)


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

    event_layers = compute_sdcmap_event_layers(driver=driver, mapped_field=mapped_field, config=config)

    lats, lons = grid_coordinates(mapped_field)

    npz_path = save_layers_npz(args.out_dir / "sdcmap_layers.npz", layers=event_layers, lats=lats, lons=lons)

    coastline = load_coastline(paths["coastline"])
    positive_png_path = plot_correlation_maps_by_lag(
        lag_maps=event_layers["positive"]["lag_maps"],
        lats=lats,
        lons=lons,
        coastline=coastline,
        out_path=args.out_dir / "sdcmap_layers_positive.png",
        title="SDCMap positive-event lagged correlation maps",
    )
    negative_png_path = plot_correlation_maps_by_lag(
        lag_maps=event_layers["negative"]["lag_maps"],
        lats=lats,
        lons=lons,
        coastline=coastline,
        out_path=args.out_dir / "sdcmap_layers_negative.png",
        title="SDCMap negative-event lagged correlation maps",
    )
    positive_compact_png_path = plot_layer_maps_compact(
        layers=event_layers["positive"]["layers"],
        lats=lats,
        lons=lons,
        coastline=coastline,
        out_path=args.out_dir / "sdcmap_layers_positive_compact.png",
        title="Positive events · A/B/C/D",
    )
    negative_compact_png_path = plot_layer_maps_compact(
        layers=event_layers["negative"]["layers"],
        lats=lats,
        lons=lons,
        coastline=coastline,
        out_path=args.out_dir / "sdcmap_layers_negative_compact.png",
        title="Negative events · A/B/C/D",
    )
    static_layer_keys = ("corr_mean", "driver_rel_time_mean", "lag_mean", "timing_combo")
    static_outputs = []
    for sign_key in ("positive", "negative"):
        for layer_key in static_layer_keys:
            static_outputs.append(
                plot_single_layer_map(
                    layers=event_layers[sign_key]["layers"],
                    layer_key=layer_key,
                    lats=lats,
                    lons=lons,
                    coastline=coastline,
                    out_path=args.out_dir / f"sdcmap_{sign_key}_{layer_key}.png",
                    title=f"{sign_key.title()} · {layer_key.replace('_mean', '').replace('_', ' ')}",
                )
            )

    print(f"Driver dataset: {args.driver_dataset}")
    print(f"Field dataset: {args.field_dataset}")
    print(f"Saved: {npz_path}")
    print(f"Saved: {positive_png_path}")
    print(f"Saved: {negative_png_path}")
    print(f"Saved: {positive_compact_png_path}")
    print(f"Saved: {negative_compact_png_path}")
    for path in static_outputs:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
