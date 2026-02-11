# Interpreting the 4 SDCMap Output Layers

This note explains the four variables shown in the compact `2x2` map figure produced by `plot_layer_maps_compact`:

- `corr_mean`
- `lag_mean`
- `driver_rel_time_mean`
- `dominant_sign`

All four are computed **per grid cell** from SDC pairs returned by `compute_sdc(...)` in `src/sdcpy_map/layers.py`.

## 1) How each grid-cell summary is built

For each `(lat, lon)` cell:

1. Compute all SDC fragment-pairs between:
   - `ts1`: driver series (for example Niño3.4)
   - `ts2`: local mapped-variable anomaly series
2. Keep only finite and significant pairs (`p_value <= alpha`).
   - In the current package defaults, significance comes from a **one-tailed** permutation test (`SDCMapConfig.two_tailed=False`).
3. Split pairs by sign:
   - positive: `r > 0`
   - negative: `r < 0`
4. Keep only the top fraction in each sign:
   - `cpos = floor(len(pos) * top_fraction)`
   - `cneg = floor(len(neg) * top_fraction)`
   - a sign is only considered if at least 2 selected pairs exist.
5. If both signs are available, choose the sign whose selected set has larger `abs(mean(r))`.
6. Compute layer values from the selected set.

If no valid selected set exists, the cell is `NaN`.

## 2) Meaning of each of the 4 plotted layers

## `corr_mean` (panel: “Mean extreme correlation”)

Definition:

`corr_mean = mean(r_selected)`

Interpretation:

- Measures average strength of the selected dominant extreme relation.
- Close to `+1`: strong positive co-variability.
- Close to `-1`: strong negative co-variability.
- Near `0`: weak relation (or mixed weak residual relation after filtering).

Plot defaults:

- Colormap: `RdBu_r`
- Fixed range: `[-1, 1]`

## `lag_mean` (panel: “Mean lag (months)”)

Definition:

`lag_mean = mean(lag_selected)`, where `lag = start_1 - start_2`

With `start_1` = driver fragment start index, `start_2` = local fragment start index:

- `lag < 0`: driver window starts earlier than local window (driver leads).
- `lag > 0`: driver window starts later than local window (local leads).
- `lag ≈ 0`: near-synchronous timing.

In the tutorial notebook, the unit is months because the data are monthly.

Plot defaults:

- Colormap: `coolwarm`
- Auto-scaled to data range

## `driver_rel_time_mean` (panel: “Mean driver-relative time (months)”)

Definition:

`driver_rel_start = start_1 - peak_idx`

`driver_rel_time_mean = mean(driver_rel_start_selected)`

Interpretation:

- Tells when the selected strong relationship windows occur relative to the reference driver peak month.
- Negative values: relationship tends to occur **before** the peak.
- Positive values: relationship tends to occur **after** the peak.
- Around `0`: concentrated around the peak.

Plot defaults:

- Colormap: `PuOr`
- Auto-scaled to data range

## `dominant_sign` (panel: “Dominant sign (+1/-1)”)

Definition:

- `+1` if selected dominant set is positive-correlation extreme set.
- `-1` if selected dominant set is negative-correlation extreme set.

Interpretation:

- Encodes polarity only, not magnitude.
- Useful as a quick phase/polarity map.

Plot defaults:

- Colormap: custom two-color map (`blue` for `-1`, `red` for `+1`)
- Fixed range: `[-1, 1]`

## 3) How to read the 4 panels together

- Use `corr_mean` first to find where the relationship is strongest.
- Use `dominant_sign` to identify whether that relationship is in-phase (`+`) or anti-phase (`-`).
- Use `lag_mean` to infer lead/lag direction.
- Use `driver_rel_time_mean` to place the relationship in event-relative time (pre-peak vs post-peak).

Combined, they answer: **where**, **how strong**, **with what sign**, and **when** the strongest local driver coupling tends to appear.

## 4) Important caveats

- Results depend on `fragment_size`, lag bounds, `alpha`, and `top_fraction`.
- `lag_mean` sign convention follows `lag = start_1 - start_2` in `sdcpy`.
- Auto-scaled colorbars for `lag_mean` and `driver_rel_time_mean` can change from run to run; compare ranges before comparing maps across experiments.
