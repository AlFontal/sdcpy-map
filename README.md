# sdcpy-map

Spatial mapping utilities for Scale-Dependent Correlation (SDC), built as an extension package on top of [`sdcpy`](https://github.com/AlFontal/sdcpy).

The main product is a compact **4-layer SDC map** summarizing:

1. mean extreme correlation (`corr_mean`),
2. mean lag (`lag_mean`),
3. mean driver-relative timing (`driver_rel_time_mean`),
4. dominant sign (`dominant_sign`).

## Install

```bash
pip install -e .
```

## Default datasets (exchangeable)

`sdcpy-map` ships dataset registries so the driver index and mapped variable can be swapped without changing the core workflow.

Default demo pair:

- Driver: `pdo` (NOAA PSL Pacific Decadal Oscillation index)
- Field: `ncep_air` (NOAA NCEP/NCAR monthly near-surface air temperature)

Other bundled keys:

- Drivers: `pdo`, `nao`, `nino34`
- Fields: `ncep_air`, `ersstv5_sst`, `oisst_v2_sst`

## Short tutorial (main 4-layer product)

```python
from pathlib import Path

from sdcpy_map import (
    SDCMapConfig,
    align_driver_to_field,
    compute_sdcmap_layers,
    fetch_public_example_data,
    grid_coordinates,
    load_coastline,
    load_driver_series,
    load_field_anomaly_subset,
    plot_layer_maps_compact,
    save_layers_npz,
)

data_dir = Path(".data")
out_dir = Path(".output")
out_dir.mkdir(exist_ok=True)

config = SDCMapConfig(
    # one-tailed significance by default in sdcpy-map:
    two_tailed=False,
    fragment_size=12,
    n_permutations=49,
    min_lag=-6,
    max_lag=6,
    alpha=0.05,
    top_fraction=0.25,
    # choose a window/domain appropriate for the chosen driver+field:
    time_start="2010-01-01",
    time_end="2023-12-01",
    peak_date="2015-01-01",
    lat_min=20,
    lat_max=70,
    lon_min=-160,
    lon_max=-60,
)

paths = fetch_public_example_data(
    data_dir=data_dir,
    driver_key="pdo",
    field_key="ncep_air",
)

driver = load_driver_series(paths["driver"], config=config, driver_key="pdo")
mapped_field = load_field_anomaly_subset(paths["field"], config=config, field_key="ncep_air")
driver = align_driver_to_field(driver, mapped_field)

layers = compute_sdcmap_layers(driver=driver, mapped_field=mapped_field, config=config)
lats, lons = grid_coordinates(mapped_field)
coastline = load_coastline(paths["coastline"])

save_layers_npz(out_dir / "sdcmap_layers.npz", layers=layers, lats=lats, lons=lons)
fig, axes, cbar_axes = plot_layer_maps_compact(
    layers=layers,
    lats=lats,
    lons=lons,
    coastline=coastline,
    out_path=out_dir / "sdcmap_layers.png",
    title="SDCMap-style layers (PDO vs NCEP air anomalies)",
    return_handles=True,  # subplot handles available for custom edits
)
```

## Driver and mapped-variable exchangeability

The SDC map computation itself is already generic:

- `compute_sdcmap_layers(driver=<pd.Series>, mapped_field=<xr.DataArray>, config=...)`

So you can pass any driver series and any mapped field as long as:

1. they share the same monthly timestamps after alignment,
2. the field has dimensions `(time, lat, lon)`.

Example with bundled alternatives:

```python
paths = fetch_public_example_data(data_dir, driver_key="nao", field_key="ersstv5_sst")
driver = load_driver_series(paths["driver"], config, driver_key="nao")
mapped_field = load_field_anomaly_subset(paths["field"], config, field_key="ersstv5_sst")
driver = align_driver_to_field(driver, mapped_field)
```

## CLI demo

```bash
sdcpy-map-demo --data-dir .data --out-dir .output --driver-dataset pdo --field-dataset ncep_air
```

Outputs:

- `.output/sdcmap_layers.npz`
- `.output/sdcmap_layers.png`

## In-depth example

- Notebook: `reports/index.ipynb`
- Rendered HTML: `reports/_site/index.html`
- Quarto config: `reports/_quarto.yml`

Render locally:

```bash
quarto render reports --execute
```

## Development

```bash
pip install -e .[dev]
pytest -q
```
