# Open SDCMap Algorithm Questions

These are key decisions we should align on before stabilizing a public API and integrating into `sdcpy` core.

## 1) Event framing and reference time

1. Should SDCMap always be event-based (start/peak windows), or should continuous mode be first-class?
2. If event-based, what is the canonical reference: onset, peak, minimum/maximum, threshold crossing, or user-specified date?
3. For non-ENSO drivers, what metadata schema should define events?

## 2) Correlation significance and permutation strategy

1. Should significance be one-sided by sign (as in legacy MATLAB) or two-sided by default?
2. Should permutation shuffle individual points, or use block/circular permutations to preserve autocorrelation?
3. What is the default `n_permutations` for publication-quality vs exploratory runs?
4. Should we expose exact p-value methods (Monte Carlo, analytical, both)?

## 3) Extreme selection and dominance logic

1. Is the top-fraction rule fixed at 25% or configurable by default?
2. Is the minimum selected count threshold (`>=2`) scientifically justified, or should it depend on sample size?
3. When both signs are present, should dominance be chosen by absolute mean correlation magnitude, count, or significance-weighted metric?
4. Do we want to preserve both-sign maps rather than only dominant-sign maps?

## 4) Lag convention and indexing

1. What is the canonical lag sign convention in docs/API (positive = driver leads or lags)?
2. Should lag be reported in index steps or physical units (months/days) by default?
3. How do we guarantee MATLAB-equivalent index conversion (1-based legacy) in Python outputs?

## 5) Preprocessing assumptions

1. Should anomaly computation be mandatory (seasonal cycle removal), optional, or user-owned?
2. Should detrending be part of default preprocessing?
3. How should we handle missing values: pairwise drop, interpolation, masked output?

## 6) Multiple comparisons and robustness

1. Do we need multiple-testing control across grid cells/lags/windows (e.g., FDR)?
2. Should we report uncertainty intervals for layer summaries?
3. What robustness diagnostics are required before publishing maps?

## 7) Output contract and interoperability

1. Which layers are required in the public API contract?
2. Should output be NPZ only, or also NetCDF/Zarr?
3. Should metadata (driver name, config, data URLs, software versions) be embedded in outputs by default?

## 8) Performance and scale

1. What runtime target should we optimize for (regional vs global runs)?
2. Should we support optional parallel execution by lat/lon chunks in API?
3. Do we need chunked/dask execution for larger daily high-resolution SST products?
