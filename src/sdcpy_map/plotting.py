"""Plotting utilities for SDCMap-style outputs."""

from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

STATIC_LAYER_SPECS = (
    ("corr_mean", "A. Correlation", "RdBu_r", -1.0, 1.0),
    ("driver_rel_time_mean", "B. Position", "PuOr", None, None),
    ("lag_mean", "C. Lag", "coolwarm", None, None),
    ("timing_combo", "D. Timing", "BrBG", None, None),
)


def _plot_coastline(ax: Axes, coastline: gpd.GeoDataFrame, *, linewidth: float) -> None:
    if coastline is None or getattr(coastline, "empty", True):
        return

    geom_types = coastline.geom_type.astype(str) if hasattr(coastline, "geom_type") else []
    polygon_mask = geom_types.isin(["Polygon", "MultiPolygon"]) if len(geom_types) else []
    if len(polygon_mask):
        polygon_geoms = coastline.loc[polygon_mask]
        line_geoms = coastline.loc[~polygon_mask]
    else:
        polygon_geoms = coastline.iloc[0:0]
        line_geoms = coastline

    if len(polygon_geoms):
        polygon_geoms.boundary.plot(ax=ax, color="black", linewidth=linewidth, zorder=3)
    if len(line_geoms):
        line_geoms.plot(ax=ax, color="black", linewidth=linewidth, zorder=3)


def plot_layer_maps_compact(
    layers: dict[str, np.ndarray],
    lats: np.ndarray,
    lons: np.ndarray,
    coastline: gpd.GeoDataFrame,
    out_path: Path | str | None = None,
    title: str = "SDCMap-style layers (driver vs mapped-variable anomalies)",
    return_handles: bool = False,
) -> Path | tuple[Figure, np.ndarray, list[Axes]] | None:
    """Render compact 2x2 layer panels with full-height per-panel colorbars.

    When ``return_handles=True``, returns ``(fig, axes, colorbar_axes)`` so callers
    can directly access subplot artists. Otherwise, the figure is closed and the
    saved output path (or ``None`` if ``out_path`` is not provided) is returned.
    """
    save_path: Path | None = None
    if out_path is not None:
        save_path = Path(out_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    mono_font = ["DejaVu Sans Mono", "Liberation Mono", "Courier New", "monospace"]

    with plt.rc_context({"font.family": "monospace", "font.monospace": mono_font}):
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(14, 7.6),
            sharex=True,
            sharey=True,
            constrained_layout=False,
        )

        layer_defs = STATIC_LAYER_SPECS
        cbar_axes: list[Axes] = []

        for idx, (ax, (key, subplot_title, cmap, vmin, vmax)) in enumerate(zip(axes.ravel(), layer_defs)):
            row, col = divmod(idx, 2)

            mesh = ax.pcolormesh(lons, lats, layers[key], shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            _plot_coastline(ax, coastline, linewidth=0.45)

            ax.set_title(subplot_title, fontsize=10)
            ax.set_xlim(float(np.nanmin(lons)), float(np.nanmax(lons)))
            ax.set_ylim(float(np.nanmin(lats)), float(np.nanmax(lats)))

            if row == 0:
                ax.tick_params(axis="x", which="both", labelbottom=False, bottom=False)
                ax.set_xlabel("")
            else:
                ax.set_xlabel("Longitude", fontsize=9)
                ax.tick_params(axis="x", labelsize=8)

            if col == 0:
                ax.set_ylabel("Latitude", fontsize=9)
                ax.tick_params(axis="y", labelsize=8)
            else:
                ax.tick_params(axis="y", which="both", labelleft=False, left=False)
                ax.set_ylabel("")

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="2.6%", pad=0.035)
            cbar = fig.colorbar(mesh, cax=cax)
            cbar.ax.tick_params(labelsize=8)
            cbar_axes.append(cax)

        fig.subplots_adjust(
            left=0.052,
            right=0.996,
            bottom=0.075,
            top=0.90,
            wspace=0.02,
            hspace=0.06,
        )
        fig.suptitle(title, fontsize=12)
        if save_path is not None:
            fig.savefig(save_path, dpi=180, bbox_inches="tight", pad_inches=0.02)

        if return_handles:
            return fig, axes, cbar_axes

        plt.close(fig)
        return save_path

    return None


def plot_single_layer_map(
    layers: dict[str, np.ndarray],
    layer_key: str,
    lats: np.ndarray,
    lons: np.ndarray,
    coastline: gpd.GeoDataFrame,
    out_path: Path | str | None = None,
    title: str | None = None,
    return_handles: bool = False,
) -> Path | tuple[Figure, Axes] | None:
    """Render one static SDC-map summary layer."""
    save_path: Path | None = None
    if out_path is not None:
        save_path = Path(out_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    spec_lookup = {key: (label, cmap, vmin, vmax) for key, label, cmap, vmin, vmax in STATIC_LAYER_SPECS}
    if layer_key not in spec_lookup:
        supported = ", ".join(key for key, *_ in STATIC_LAYER_SPECS)
        raise ValueError(f"Unsupported layer '{layer_key}'. Supported static layers: {supported}.")
    label, cmap, vmin, vmax = spec_lookup[layer_key]

    mono_font = ["DejaVu Sans Mono", "Liberation Mono", "Courier New", "monospace"]
    with plt.rc_context({"font.family": "monospace", "font.monospace": mono_font}):
        fig, ax = plt.subplots(1, 1, figsize=(8.4, 4.8), constrained_layout=False)
        mesh = ax.pcolormesh(lons, lats, layers[layer_key], shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        _plot_coastline(ax, coastline, linewidth=0.5)
        ax.set_title(title or label, fontsize=12)
        ax.set_xlabel("Longitude", fontsize=10)
        ax.set_ylabel("Latitude", fontsize=10)
        ax.tick_params(axis="both", labelsize=9)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="6%", pad=0.42)
        cbar = fig.colorbar(mesh, cax=cax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=8)
        fig.subplots_adjust(left=0.08, right=0.99, bottom=0.16, top=0.90)
        if save_path is not None:
            fig.savefig(save_path, dpi=180, bbox_inches="tight", pad_inches=0.02)

        if return_handles:
            return fig, ax

        plt.close(fig)
        return save_path

    return None


def plot_correlation_maps_by_lag(
    lag_maps: dict[str, object],
    lats: np.ndarray,
    lons: np.ndarray,
    coastline: gpd.GeoDataFrame,
    out_path: Path | str | None = None,
    title: str = "SDCMap lagged correlation maps",
    return_handles: bool = False,
    ncols: int = 4,
) -> Path | tuple[Figure, np.ndarray] | None:
    """Render one correlation map per lag for a single event class."""
    save_path: Path | None = None
    if out_path is not None:
        save_path = Path(out_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    lag_values = np.asarray(lag_maps.get("lags") or [], dtype=int)
    corr_by_lag = np.asarray(lag_maps.get("corr_by_lag"), dtype=float)
    if corr_by_lag.ndim != 3 or corr_by_lag.shape[0] != len(lag_values):
        raise ValueError("`lag_maps` must contain `lags` and `corr_by_lag` with shape (lag, lat, lon).")

    ncols = max(1, int(ncols))
    n_panels = max(1, len(lag_values))
    nrows = math.ceil(n_panels / ncols)

    mono_font = ["DejaVu Sans Mono", "Liberation Mono", "Courier New", "monospace"]
    with plt.rc_context({"font.family": "monospace", "font.monospace": mono_font}):
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4.2 * ncols, 3.1 * nrows),
            sharex=True,
            sharey=True,
            constrained_layout=False,
        )
        axes_array = np.atleast_2d(np.asarray(axes, dtype=object))
        flat_axes = axes_array.ravel()
        meshes = []

        for idx, ax in enumerate(flat_axes):
            if idx >= len(lag_values):
                ax.axis("off")
                continue
            lag = int(lag_values[idx])
            mesh = ax.pcolormesh(
                lons,
                lats,
                corr_by_lag[idx],
                shading="auto",
                cmap="RdBu_r",
                vmin=-1.0,
                vmax=1.0,
            )
            _plot_coastline(ax, coastline, linewidth=0.45)
            meshes.append(mesh)
            ax.set_title(f"Lag {lag:+d}", fontsize=10)
            ax.set_xlim(float(np.nanmin(lons)), float(np.nanmax(lons)))
            ax.set_ylim(float(np.nanmin(lats)), float(np.nanmax(lats)))
            row, col = divmod(idx, ncols)
            if row == nrows - 1:
                ax.set_xlabel("Longitude", fontsize=9)
                ax.tick_params(axis="x", labelsize=8)
            else:
                ax.tick_params(axis="x", which="both", labelbottom=False, bottom=False)
            if col == 0:
                ax.set_ylabel("Latitude", fontsize=9)
                ax.tick_params(axis="y", labelsize=8)
            else:
                ax.tick_params(axis="y", which="both", labelleft=False, left=False)

        active_mesh = meshes[0] if meshes else None
        if active_mesh is not None:
            cbar = fig.colorbar(
                active_mesh,
                ax=list(flat_axes[: len(lag_values)]),
                orientation="horizontal",
                fraction=0.05,
                pad=0.06,
                aspect=40,
            )
            cbar.set_label("Correlation", fontsize=9)
            cbar.ax.tick_params(labelsize=8)

        fig.subplots_adjust(left=0.055, right=0.995, bottom=0.10, top=0.90, wspace=0.04, hspace=0.12)
        fig.suptitle(title, fontsize=12)
        if save_path is not None:
            fig.savefig(save_path, dpi=180, bbox_inches="tight", pad_inches=0.02)

        if return_handles:
            return fig, axes_array

        plt.close(fig)
        return save_path

    return None
