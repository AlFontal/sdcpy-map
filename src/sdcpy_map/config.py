"""Configuration models for sdcpy-map."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SDCMapConfig:
    """Parameters for SDCMap-style layer computation."""

    fragment_size: int = 12
    n_permutations: int = 49
    two_tailed: bool = False
    min_lag: int = -6
    max_lag: int = 6
    alpha: float = 0.05
    top_fraction: float = 0.25
    peak_date: str = "2015-11-01"

    time_start: str = "2014-01-01"
    time_end: str = "2018-12-01"

    lat_min: float = -20
    lat_max: float = 20
    lon_min: float = -170
    lon_max: float = -70
    lat_stride: int = 1
    lon_stride: int = 1
