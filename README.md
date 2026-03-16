# sdcpy-map

[![CI](https://github.com/AlFontal/sdcpy-map/actions/workflows/ci.yml/badge.svg)](https://github.com/AlFontal/sdcpy-map/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.9%20to%203.12-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

`sdcpy-map` computes event-conditioned spatial Scale-Dependent Correlation (SDC) maps from a scalar driver time series and a gridded field.

Typical use case:
- driver: PDO, ENSO, NAO, or another climate index
- field: SST, air temperature, precipitation, or another `(time, lat, lon)` anomaly field

The method is designed to answer a scientific question of the form:

> For the strongest positive and negative events in a driver series, where does the field respond, how strongly, and with what timing?

## Install

```bash
uv sync
```

## Method overview

The workflow is event-based rather than full-series averaged.

1. Detect positive and negative extrema in the driver separately.
2. Select up to `N+` positive events and `N-` negative events, enforcing minimum separation.
3. Define a base-state threshold from the weakest selected event magnitude.
4. Build a baseline field from time steps that remain inside that threshold.
5. For each selected event and each grid cell, compare a centered driver window against lagged field windows.
6. Keep the strongest significant lagged correlation for that event/cell pair.
7. Average the retained event-level responses within the positive and negative classes.

## Core definitions

Let the selected event amplitudes be `|x_1|, |x_2|, ..., |x_k|`. Define

$$
x_0 = \min(|x_1|, |x_2|, \dots, |x_k|)
$$

and the base-state threshold

$$
\tau = \beta x_0
$$

where `beta` is the user-controlled base-state factor.

Baseline time steps satisfy

$$
|x(t)| < \tau
$$

after excluding the selected event windows themselves.

For each event and lag, the local SDC correlation is computed between:

- a centered driver segment of width `r_w`
- a field segment of the same width shifted by lag `\ell`

The method keeps the strongest significant correlation at each cell for each event, then averages those retained event-level results by sign class.

## Outputs

The top-level result contains three main blocks:

- `result["positive"]`
- `result["negative"]`
- `result["event_catalog"]`

Each event class contains:

- `layers`: static A/B/C/D summary layers
- `lag_maps`: lag-resolved correlation maps
- `valid_cells_by_lag`: number of valid cells at each lag

The static A/B/C/D layers summarize:

- `A`: maximum retained SDC correlation
- `B`: response position relative to the driver peak
- `C`: lag of the retained response
- `D`: combined timing relative to the driver peak

## Bundled datasets

Default demo pair:

- Driver: `pdo`
- Field: `ncep_air`

Other bundled dataset keys:

- Drivers: `pdo`, `nao`, `nino34`
- Fields: `ncep_air`, `ersstv5_sst`, `oisst_v2_sst`

## Quick start

```python
from pathlib import Path

from sdcpy_map import (
    SDCMapConfig,
    align_driver_to_field,
    compute_sdcmap_event_layers,
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
    two_tailed=False,
    correlation_width=12,
    n_positive_peaks=3,
    n_negative_peaks=3,
    base_state_beta=0.5,
    n_permutations=49,
    min_lag=-6,
    max_lag=6,
    alpha=0.05,
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

result = compute_sdcmap_event_layers(driver=driver, mapped_field=mapped_field, config=config)
lats, lons = grid_coordinates(mapped_field)
coastline = load_coastline(paths["coastline"])

save_layers_npz(out_dir / "sdcmap_layers.npz", layers=result, lats=lats, lons=lons)

plot_correlation_maps_by_lag(
    lag_maps=result["positive"]["lag_maps"],
    lats=lats,
    lons=lons,
    coastline=coastline,
    out_path=out_dir / "sdcmap_positive_by_lag.png",
    title="Positive-event SDC map by lag",
)

plot_layer_maps_compact(
    layers=result["positive"]["layers"],
    lats=lats,
    lons=lons,
    coastline=coastline,
    out_path=out_dir / "sdcmap_positive_summary.png",
    title="Positive-event static A/B/C/D summary",
)
```

## Run the example

```bash
uv run python examples/public_enso_sst_demo.py
```

Or use the CLI:

```bash
uv run sdcpy-map-demo --data-dir .data --out-dir .output --driver-dataset pdo --field-dataset ncep_air
```

## Development

```bash
uv sync --extra dev
uv run pytest -q
```
