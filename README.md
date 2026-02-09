# sdcpy-map

Spatial mapping helpers for Scale-Dependent Correlation (SDC), built as an extension package on top of [`sdcpy`](https://github.com/AlFontal/sdcpy).

## Why this package

`sdcpy` provides core SDC computation. `sdcpy-map` adds opinionated utilities for:

- downloading common public driver/field datasets for demos,
- computing SDCMap-style spatial summary layers,
- generating compact map layouts for interpretation,
- providing a path that can eventually be integrated into the main `sdcpy` package.

## Install

```bash
pip install -e .
```

## Quick demo

```bash
sdcpy-map-demo --data-dir .data --out-dir .output
```

This writes:

- `sdcmap_layers.npz`
- `sdcmap_layers.png`

## Public demo data sources

- Nino3.4 monthly anomalies (NOAA PSL):
  - `https://psl.noaa.gov/data/correlation/nina34.anom.csv`
- ERSSTv5 sample netCDF (xarray-data mirror):
  - `https://raw.githubusercontent.com/pydata/xarray-data/master/ersstv5.nc`
- Natural Earth coastline shapefile (110m):
  - `https://naciscdn.org/naturalearth/110m/physical/ne_110m_coastline.zip`

## Development

```bash
pytest -q
```
