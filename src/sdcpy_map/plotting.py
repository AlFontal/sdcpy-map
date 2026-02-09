"""Plotting utilities for SDCMap-style outputs."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_layer_maps_compact(
    layers: dict[str, np.ndarray],
    lats: np.ndarray,
    lons: np.ndarray,
    coastline: gpd.GeoDataFrame,
    out_path: Path | str,
    title: str = "SDCMap-style layers (driver vs SST anomalies)",
) -> Path:
    """Render compact 2x2 layer panels with full-height per-panel colorbars."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

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

        layer_defs = [
            ("corr_mean", "Mean extreme correlation", "RdBu_r", -1.0, 1.0),
            ("lag_mean", "Mean lag (months)", "coolwarm", None, None),
            ("driver_rel_time_mean", "Mean driver-relative time (months)", "PuOr", None, None),
            ("dominant_sign", "Dominant sign (+1/-1)", ListedColormap(["#2166AC", "#B2182B"]), -1, 1),
        ]

        for idx, (ax, (key, subplot_title, cmap, vmin, vmax)) in enumerate(zip(axes.ravel(), layer_defs)):
            row, col = divmod(idx, 2)

            mesh = ax.pcolormesh(lons, lats, layers[key], shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            coastline.plot(ax=ax, color="black", linewidth=0.4)

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
            cax = divider.append_axes("right", size="2.8%", pad=0.05)
            cbar = fig.colorbar(mesh, cax=cax)
            cbar.ax.tick_params(labelsize=8)

        fig.subplots_adjust(left=0.06, right=0.985, bottom=0.085, top=0.88, wspace=0.22, hspace=0.22)
        fig.suptitle(title, fontsize=12)
        fig.savefig(out_path, dpi=180)
        plt.close(fig)

    return out_path
