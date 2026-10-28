# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
"""Tests for configuration parsing."""

from pathlib import Path

import pytest

from spidr.config import (
    MaskingConfig,
    SpidRConfig,
    read_config,
)


@pytest.mark.parametrize("prob", [-0.1, 1.1])
def test_masking_config_rejects_probabilities_outside_unit_interval(prob: float) -> None:
    with pytest.raises(ValueError, match="mask_prob must be in"):
        MaskingConfig(mask_prob=prob)
    with pytest.raises(ValueError, match="mask_channel_prob must be in"):
        MaskingConfig(mask_channel_prob=prob)


def test_read_example_config() -> None:
    cfg = read_config(Path(__file__).parents[1] / "configs" / "example.toml")
    assert cfg.run.model_type == "spidr"
    assert isinstance(cfg.model, SpidRConfig)
    assert set(cfg.validation) == {"dev-clean", "dev-other"}
    assert cfg.optimizer.mixed_precision


def test_read_config_rejects_unknown_format(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("run: {}")
    with pytest.raises(ValueError, match="Unsupported config file format"):
        read_config(path)
