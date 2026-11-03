# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
"""Tests for configuration parsing."""

from pathlib import Path
from typing import get_args

import pytest

from spidr.config import (
    DinoSRConfig,
    MaskingConfig,
    ModelType,
    RunConfig,
    SpidRConfig,
    model_config_class,
    read_config,
)


@pytest.mark.parametrize("prob", [-0.1, 1.1])
def test_masking_config_rejects_probabilities_outside_unit_interval(prob: float) -> None:
    with pytest.raises(ValueError, match="mask_prob must be in"):
        MaskingConfig(mask_prob=prob)
    with pytest.raises(ValueError, match="mask_channel_prob must be in"):
        MaskingConfig(mask_channel_prob=prob)


@pytest.mark.parametrize("model_type", get_args(ModelType))
def test_every_model_type_has_a_config_class(model_type: ModelType) -> None:
    """Adding a name to `ModelType` without registering its config must fail here, not at training time."""
    assert issubclass(model_config_class(model_type), DinoSRConfig)


def test_model_config_class_rejects_unknown_model_type() -> None:
    with pytest.raises(ValueError, match="Unknown model type 'wav2vec2', expected one of"):
        model_config_class("wav2vec2")  # ty: ignore[invalid-argument-type]


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


def test_wandb_id_is_stable_for_a_run_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = RunConfig(workdir=str(tmp_path), wandb_project="project", wandb_name="name")
    assert len(run.wandb_id) == 8
    assert set(run.wandb_id) <= set("abcdefghijklmnopqrstuvwxyz0123456789")
    monkeypatch.chdir(tmp_path)
    assert RunConfig(workdir=".", wandb_project="project", wandb_name="name").wandb_id == run.wandb_id
    assert not run.dir.exists()


def test_wandb_id_differs_across_run_directories(tmp_path: Path) -> None:
    ids = {
        RunConfig(workdir=str(tmp_path / workdir), wandb_project=project, wandb_name=name).wandb_id
        for workdir, project, name in [("a", "p", "n"), ("b", "p", "n"), ("a", "q", "n"), ("a", "p", "m")]
    }
    assert len(ids) == 4
