# Interpreting the 4 Compact SDCMap Layers

This note explains the four variables shown in the compact `2x2` compatibility figure produced by `plot_layer_maps_compact`.

The canonical SDC-map figure is now the lag-resolved correlation panel set produced by `plot_correlation_maps_by_lag`. This note only documents the legacy compact summary that is still emitted for transition/debugging purposes.

- `corr_mean`
- `driver_rel_time_mean`
- `lag_mean`
- `timing_combo`

These compact layers are now derived from the event-conditioned workflow in `src/sdcpy_map/layers.py`.

## 1) How each grid-cell summary is built

For each `(lat, lon)` cell and for each selected driver event:

1. Build a centered driver window of width `correlation_width` around the event peak.
2. For each lag in `[min_lag, max_lag]`, extract the equally sized field window implied by `lag = start_1 - start_2`.
3. Compute the local correlation for that event/lag pair.
4. Estimate significance with the configured permutation test.
5. Keep the strongest significant lag for that event at that grid cell.
6. Average those event-level summaries across the selected events of the class.

So the compact view is not built from full-series SDC fragments anymore. It is built from the best significant event-local lag per selected peak, then averaged across peaks.

If no selected event yields a significant local correlation at a cell, that cell is `NaN`.

## 2) Meaning of each of the 4 plotted layers

## `corr_mean` (panel: “Mean extreme correlation”)

Definition:

`corr_mean = mean(r_best_event)`

where `r_best_event` is the strongest significant lagged correlation retained for each selected event at that cell.

Interpretation:

- Close to `+1`: the field tends to co-vary positively with the selected event windows.
- Close to `-1`: the field tends to vary oppositely during those event windows.
- Near `0`: weak or inconsistent event-conditioned relation.

Plot defaults:

- Colormap: `RdBu_r`
- Fixed range: `[-1, 1]`

## `driver_rel_time_mean` (panel: “B. Peak-relative position”)

Definition:

`driver_rel_time_mean = mean(center_best_event - peak_idx)`

Interpretation:

- This is the position of the retained driver-fragment center relative to the selected event peak.
- Negative values mean the retained driver fragment is centered before the peak.
- Positive values mean it is centered after the peak.
- Values are expressed in native data steps. For monthly inputs, these are months.

Plot defaults:

- Colormap: `PuOr`
- Auto-scaled to data range

## `lag_mean` (panel: “C. Lag”)

Definition:

`lag_mean = mean(lag_best_event)`, where `lag = start_1 - start_2`

Interpretation:

- `lag < 0`: the driver fragment is earlier than the field fragment (driver leads).
- `lag > 0`: the driver fragment is later than the field fragment (field leads).
- `lag ≈ 0`: near-synchronous timing.
- Values are expressed in native data steps. For monthly inputs, these are months.

Plot defaults:

- Colormap: `coolwarm`
- Auto-scaled to data range

## `timing_combo` (panel: “D. Combined timing”)

Definition:

`timing_combo = driver_rel_time_mean - lag_mean`

Interpretation:

- This is the exact response timing used in the original slides: the retained field-fragment center with respect to the driver peak.
- It summarizes the total temporal placement of the retained response in one map.
- Values are expressed in native data steps. For monthly inputs, these are months.

Plot defaults:

- Colormap: `BrBG`
- Auto-scaled to data range

## 3) How to read the 4 panels together

- Use `corr_mean` first to find where the selected events have the strongest average response.
- Use `driver_rel_time_mean` to see where the retained response sits relative to the driver peak.
- Use `lag_mean` to infer whether the field tends to lead or lag the selected event windows.
- Use `timing_combo` to recover the exact combined timing described in the methodology slides.

Combined, they answer: **where**, **how strong**, **where relative to the peak**, **with what lag**, and **with what exact combined timing** the selected event classes are expressed.

## 4) Important caveats

- Results depend on `correlation_width`, lag bounds, `alpha`, and the selected `N+` / `N-`.
- `lag_mean` sign convention follows `lag = start_1 - start_2`.
- `driver_rel_time_mean` is derived from the best-scoring fragment center inside the local event neighborhood, not from the full record.
- `timing_combo` is derived from `driver_rel_time_mean - lag_mean`, so interpret it jointly with those two maps.
- Auto-scaled colorbars for `driver_rel_time_mean`, `lag_mean`, and `timing_combo` can change from run to run; compare ranges before comparing maps across experiments.
