"""Configuration models for sdcpy-map."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SDCMapConfig:
    """Parameters for event-based SDCMap computation."""

    correlation_width: int = 12
    n_positive_peaks: int = 3
    n_negative_peaks: int = 3
    base_state_beta: float = 0.5

    n_permutations: int = 49
    two_tailed: bool = False
    min_lag: int = -6
    max_lag: int = 6
    alpha: float = 0.05

    time_start: str = "2014-01-01"
    time_end: str = "2018-12-01"

    lat_min: float = -20
    lat_max: float = 20
    lon_min: float = -170
    lon_max: float = -70
    lat_stride: int = 1
    lon_stride: int = 1

    # Deprecated compatibility aliases. These remain accepted for one transition cycle.
    fragment_size: int | None = None
    top_fraction: float | None = 0.25
    peak_date: str | None = None

    def __post_init__(self) -> None:
        if self.fragment_size is not None:
            object.__setattr__(self, "correlation_width", int(self.fragment_size))
        object.__setattr__(self, "fragment_size", int(self.correlation_width))

        if int(self.correlation_width) < 2:
            raise ValueError("`correlation_width` must be >= 2.")
        if int(self.n_positive_peaks) < 0:
            raise ValueError("`n_positive_peaks` must be >= 0.")
        if int(self.n_negative_peaks) < 0:
            raise ValueError("`n_negative_peaks` must be >= 0.")
        if float(self.base_state_beta) <= 0:
            raise ValueError("`base_state_beta` must be > 0.")
        if int(self.min_lag) > int(self.max_lag):
            raise ValueError("`min_lag` must be <= `max_lag`.")
        if not 0 < float(self.alpha) < 1:
            raise ValueError("`alpha` must be between 0 and 1.")
        if self.top_fraction is not None and not 0 < float(self.top_fraction) <= 1:
            raise ValueError("`top_fraction` must be between 0 and 1 when provided.")
