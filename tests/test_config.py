# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
"""Tests for configuration parsing."""

import dataclasses
import json
from functools import partial
from pathlib import Path
from typing import Any, get_args

import pytest

from spidr.config import (
    DinoSRConfig,
    EntropySchedule,
    MaskingConfig,
    ModelType,
    RunConfig,
    SpidR2Config,
    SpidRConfig,
    model_config_class,
    model_config_from_dict,
    read_config,
)


@pytest.mark.parametrize("term", ["predictive", "codebook"])
@pytest.mark.parametrize(
    ("schedule", "timescale", "final_weight", "match"),
    [
        ("linear", 5.0, 0.001, "Unknown .* entropy schedule"),
        ("constant_then_exponential_decrease", 0.0, 0.001, "must be positive"),
        ("constant_then_exponential_decrease", 5.0, 0.5, "must be smaller"),
        ("constant_then_exponential_increase", 5.0, 0.001, "must be greater"),
    ],
)
def test_spidr2_config_rejects_inconsistent_entropy_schedules(
    term: str, schedule: EntropySchedule, timescale: float, final_weight: float, match: str
) -> None:
    """Every term is checked: a valid predictive schedule must not excuse a broken codebook one."""
    build = (
        partial(
            SpidR2Config,
            predictive_entropy_weight=0.1,
            predictive_entropy_schedule=schedule,
            predictive_entropy_timescale=timescale,
            predictive_entropy_final_weight=final_weight,
        )
        if term == "predictive"
        else partial(
            SpidR2Config,
            codebook_entropy_weight=0.1,
            codebook_entropy_schedule=schedule,
            codebook_entropy_timescale=timescale,
            codebook_entropy_final_weight=final_weight,
        )
    )
    with pytest.raises(ValueError, match=match):
        build()


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


def test_read_config_builds_the_model_config_of_the_model_type(tmp_path: Path) -> None:
    """`SpidR2Config` subclasses `SpidRConfig`, so only `model_type` can tell the two apart."""
    path = tmp_path / "config.toml"
    path.write_text(
        '[run]\nworkdir = "/tmp/workdir"\nwandb_project = "spidr"\nwandb_name = "spidr2"\nmodel_type = "spidr2"\n'
        '[data]\nmanifest = "/tmp/train.csv"\n[model]\ncodebook_entropy_weight = 0.5\n'
    )
    cfg = read_config(path)
    assert isinstance(cfg.model, SpidR2Config)
    assert cfg.model.codebook_entropy_weight == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("name", "config_class"),
    [("spidr", SpidRConfig), ("spidr2", SpidR2Config)],
)
def test_read_example_config(name: str, config_class: type[DinoSRConfig]) -> None:
    cfg = read_config(Path(__file__).parents[1] / "configs" / f"{name}.toml")
    assert cfg.run.model_type == name
    assert type(cfg.model) is config_class
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


# The `config.json` released with the SpidR 2.0 checkpoint.
RELEASED_SPIDR2_CONFIG: dict[str, Any] = {
    "run": {"model_type": "spidr"},
    "model": {
        "extractor_mode": "layer_norm",
        "extractor_conv_bias": False,
        "extractor_conv_layer_config": [[512, 10, 5]] + [[512, 3, 2]] * 4 + [[512, 2, 2]] * 2,
        "encoder_embed_dim": 768,
        "encoder_projection_dropout": 0,
        "encoder_pos_conv_kernel": 95,
        "encoder_pos_conv_groups": 16,
        "encoder_pos_conv_depth": 5,
        "encoder_num_layers": 12,
        "encoder_num_heads": 12,
        "encoder_attention_dropout": 0.1,
        "encoder_ff_interm_features": 3072,
        "encoder_ff_interm_dropout": 0.0,
        "encoder_dropout": 0.1,
        "encoder_layer_norm_first": False,
        "encoder_layer_drop": 0.0,
        "encoder_qkv_bias": False,
        "feature_grad_mult": 1.0,
        "codebook_size": 256,
        "codebook_decay": 0.9,
        "num_codebooks": 8,
        "ema_start_decay": 0.999,
        "ema_final_decay": 0.9999,
        "ema_final_step": 30000,
        "ema_exclude_layers": ["pos_conv_embed"],
        "freeze_step": 200000,
        "unfreeze_encoder_step": 0,
        "ema_timescale": 20000,
        "ema_threshold": 1e-07,
        "ema_final": 1.0,
        "teacher_use_heads": True,
        "use_soft_targets": True,
        "heads_layer_norm": "center",
        "heads_use_weight_norm": True,
        "heads_init": "sphere",
        "regularisation": {
            "reg_student": False,
            "reg_teacher": True,
            "codebook_entropy_weight": 0.1,
            "codebook_entropy_schedule": "constant_then_exponential_decrease",
            "predictive_entropy_weight": 0.1,
            "predictive_entropy_schedule": "constant",
        },
    },
}


def as_json(cfg: DinoSRConfig) -> str:
    """Compare configs as they are saved: JSON turns the tuples of `extractor_conv_layer_config` into lists."""
    return json.dumps(dataclasses.asdict(cfg))


def test_released_spidr2_config_is_the_default_config() -> None:
    """`spidr2_base` builds its model with the default config: it must be the one of the released checkpoint."""
    assert as_json(model_config_from_dict("spidr2", RELEASED_SPIDR2_CONFIG["model"])) == as_json(SpidR2Config())


def test_released_spidr2_config_keeps_its_regularisation(
    tiny_spidr2_config: SpidR2Config, tiny_released_spidr2_config: dict[str, Any]
) -> None:
    data = tiny_released_spidr2_config["model"]
    data["regularisation"] |= {"reg_student": True, "codebook_entropy_constant_steps": 123}
    expected = dataclasses.replace(tiny_spidr2_config, reg_student=True, codebook_entropy_constant_steps=123)
    assert as_json(model_config_from_dict("spidr2", data)) == as_json(expected)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("teacher_use_heads", False),
        ("use_soft_targets", False),
        ("heads_layer_norm", "affine"),
        ("heads_use_weight_norm", False),
    ],
)
def test_released_spidr2_config_rejects_another_recipe(name: str, value: object) -> None:
    with pytest.raises(ValueError, match=f"Unsupported {name}="):
        model_config_from_dict("spidr2", RELEASED_SPIDR2_CONFIG["model"] | {name: value})


def test_released_spidr2_config_rejects_another_model_type() -> None:
    with pytest.raises(ValueError, match=r"config of a SpidR 2\.0 model, but model_type is 'spidr'"):
        model_config_from_dict("spidr", RELEASED_SPIDR2_CONFIG["model"])
