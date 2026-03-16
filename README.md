# sdcpy-map

Spatial mapping utilities for event-conditioned Scale-Dependent Correlation (SDC), built on top of [`sdcpy`](https://github.com/AlFontal/sdcpy).

The canonical workflow is now:

1. detect positive and negative driver events separately,
2. define a base state from `beta * x0`,
3. exclude full selected/ignored event windows from the base-state months,
4. compute class-specific event-local lagged correlations for positive events and negative events,
5. keep the strongest significant lag per event and grid cell, then average across selected events,
6. optionally derive the old compact 4-layer output as a compatibility view.

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

## Event-conditioned tutorial

```python
from pathlib import Path

from sdcpy_map import (
    SDCMapConfig,
    align_driver_to_field,
    compute_sdcmap_event_layers,
    compute_sdcmap_layers,
    fetch_public_example_data,
    grid_coordinates,
    load_coastline,
    load_driver_series,
    load_field_anomaly_subset,
    plot_correlation_maps_by_lag,
    plot_layer_maps_compact,
    save_layers_npz,
)

data_dir = Path(".data")
out_dir = Path(".output")
out_dir.mkdir(exist_ok=True)

config = SDCMapConfig(
    # one-tailed significance by default in sdcpy-map:
    two_tailed=False,
    correlation_width=12,
    n_positive_peaks=3,
    n_negative_peaks=3,
    base_state_beta=0.5,
    n_permutations=49,
    min_lag=-6,
    max_lag=6,
    alpha=0.05,
    # choose a window/domain appropriate for the chosen driver+field:
    time_start="2010-01-01",
    time_end="2023-12-01",
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

event_result = compute_sdcmap_event_layers(driver=driver, mapped_field=mapped_field, config=config)
compact_layers = compute_sdcmap_layers(driver=driver, mapped_field=mapped_field, config=config)
lats, lons = grid_coordinates(mapped_field)
coastline = load_coastline(paths["coastline"])

save_layers_npz(out_dir / "sdcmap_layers.npz", layers=event_result, lats=lats, lons=lons)
plot_correlation_maps_by_lag(
    lag_maps=event_result["positive"]["lag_maps"],
    lats=lats,
    lons=lons,
    coastline=coastline,
    out_path=out_dir / "sdcmap_layers_positive.png",
    title="Positive-event SDC map by lag (PDO vs NCEP air anomalies)",
)

# Legacy compact summary, retained for transition/debugging only.
fig, axes, cbar_axes = plot_layer_maps_compact(
    layers=event_result["positive"]["layers"],
    lats=lats,
    lons=lons,
    coastline=coastline,
    out_path=out_dir / "sdcmap_layers_positive_compact.png",
    title="Positive-event compact compatibility view",
    return_handles=True,
)
```

Primary outputs:

- `event_result["positive"]`
- `event_result["negative"]`
- `event_result["event_catalog"]`
- `event_result["positive"]["lag_maps"]`
- `event_result["negative"]["lag_maps"]`

Method note:

- `correlation_width` defines a centered driver window around each selected event.
- `min_lag` / `max_lag` scan lagged field windows against that centered driver segment.
- each event contributes its strongest significant lag per cell,
- class layers are the average across the contributing events of that sign.

Compatibility output:

- `compute_sdcmap_layers(...)` still returns a compact combined layer set for one transition cycle.
- `plot_correlation_maps_by_lag(...)` is now the canonical renderer because the method’s primary product is a lag-resolved correlation map for each event class.

## Driver and mapped-variable exchangeability

The event-conditioned SDC map computation is generic:

- `compute_sdcmap_event_layers(driver=<pd.Series>, mapped_field=<xr.DataArray>, config=...)`

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
- `.output/sdcmap_layers_positive.png`
- `.output/sdcmap_layers_negative.png`

## Migration note
Deprecated compatibility fields still accepted in `SDCMapConfig` for one transition cycle:

- `fragment_size -> correlation_width`
- `top_fraction` and `peak_date` remain compatibility-only and are no longer part of the canonical method

New canonical controls are:

- `correlation_width`
- `n_positive_peaks`
- `n_negative_peaks`
- `base_state_beta`

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
