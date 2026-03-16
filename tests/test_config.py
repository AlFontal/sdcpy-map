from dataclasses import FrozenInstanceError

import pytest

from sdcpy_map.config import SDCMapConfig


def test_config_defaults_are_one_tailed():
    config = SDCMapConfig()
    assert config.two_tailed is False
    assert config.fragment_size == 12
    assert config.correlation_width == 12
    assert config.n_positive_peaks == 3
    assert config.n_negative_peaks == 3
    assert config.base_state_beta == 0.5


def test_config_maps_fragment_size_alias_to_correlation_width():
    config = SDCMapConfig(fragment_size=18)
    assert config.correlation_width == 18
    assert config.fragment_size == 18


def test_config_is_frozen():
    config = SDCMapConfig()
    with pytest.raises(FrozenInstanceError):
        config.alpha = 0.1
