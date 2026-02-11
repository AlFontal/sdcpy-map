from dataclasses import FrozenInstanceError

import pytest

from sdcpy_map.config import SDCMapConfig


def test_config_defaults_are_one_tailed():
    config = SDCMapConfig()
    assert config.two_tailed is False
    assert config.fragment_size == 12
    assert config.top_fraction == 0.25


def test_config_is_frozen():
    config = SDCMapConfig()
    with pytest.raises(FrozenInstanceError):
        config.alpha = 0.1
