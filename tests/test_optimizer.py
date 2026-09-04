# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
"""Tests for optimizer construction, the learning rate schedules and param group surgery."""

from typing import Literal

import pytest
import torch

from spidr.checkpoint import remove_param_group
from spidr.config import DinoSRConfig, OptimizerConfig
from spidr.models import DinoSR
from spidr.optimizer import build_optimizer, build_scheduler

Scheduler = Literal["tristage", "cosine", "rsqrt", "constant"]
Dtype = Literal["float32", "float16", "bfloat16"]
SCHEDULERS: list[Scheduler] = ["tristage", "cosine", "rsqrt", "constant"]


def test_param_groups_split_frozen_and_trainable(tiny_dinosr_config: DinoSRConfig) -> None:
    model = DinoSR(tiny_dinosr_config)
    cfg = OptimizerConfig(dtype="float32")
    optimizer, _, _ = build_optimizer(model, cfg)
    assert len(optimizer.param_groups) == 2
    trainable, freezable = (len(group["params"]) for group in optimizer.param_groups)
    expected_freezable = len(list(model.feature_extractor.parameters())) + len(
        list(model.feature_projection.parameters())
    )
    assert freezable == expected_freezable
    excluded = sum(1 for name, _ in model.named_parameters() if name.startswith("teacher"))
    assert trainable + freezable + excluded == len(list(model.parameters()))


def test_teacher_is_excluded_from_the_optimizer(tiny_dinosr_config: DinoSRConfig) -> None:
    model = DinoSR(tiny_dinosr_config)
    optimizer, _, _ = build_optimizer(model, OptimizerConfig(dtype="float32"))
    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert not any(id(p) in optimized for p in model.teacher.parameters())


def test_build_optimizer_rejects_unknown_dtype(tiny_dinosr_config: DinoSRConfig) -> None:
    model = DinoSR(tiny_dinosr_config)
    with pytest.raises(ValueError, match="float64"):
        build_optimizer(model, OptimizerConfig(dtype="float64"))  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("scheduler", SCHEDULERS)
def test_schedulers_stay_positive_and_finite(scheduler: Scheduler, tiny_dinosr_config: DinoSRConfig) -> None:
    model = DinoSR(tiny_dinosr_config)
    cfg = OptimizerConfig(
        scheduler=scheduler, warmup_steps=10, hold_steps=10, decay_steps=10, max_steps=40, dtype="float32"
    )
    optimizer, _, lr_scheduler = build_optimizer(model, cfg)
    seen = []
    for _ in range(cfg.max_steps):
        seen.append(lr_scheduler.get_last_lr()[0])
        optimizer.step()
        lr_scheduler.step()
    assert all(lr > 0 for lr in seen)
    assert all(lr == lr for lr in seen)  # No NaN.  # noqa: PLR0124
    assert seen[0] < seen[cfg.warmup_steps]  # Warmup increases the learning rate.


def test_build_scheduler_rejects_unknown_name(tiny_dinosr_config: DinoSRConfig) -> None:
    model = DinoSR(tiny_dinosr_config)
    optimizer, _, _ = build_optimizer(model, OptimizerConfig(dtype="float32"))
    with pytest.raises(ValueError, match="Unknown scheduler"):
        build_scheduler(optimizer, OptimizerConfig(scheduler="nope"))  # ty: ignore[invalid-argument-type]


def test_remove_param_group_keeps_the_scheduler_usable(tiny_dinosr_config: DinoSRConfig) -> None:
    """The freeze step drops the extractor group mid-run; the schedule must survive it."""
    model = DinoSR(tiny_dinosr_config)
    optimizer, _, scheduler = build_optimizer(model, OptimizerConfig(dtype="float32"))
    for param in optimizer.param_groups[1]["params"]:
        param.grad = torch.zeros_like(param)
    optimizer.step()
    assert any(param in optimizer.state for param in optimizer.param_groups[1]["params"])

    remove_param_group(optimizer, scheduler, 1)
    assert len(optimizer.param_groups) == 1
    assert not any(param in optimizer.state for param in model.feature_extractor.parameters())
    scheduler.step()  # The schedule must follow the surgery instead of raising or keeping two rates.
    optimizer.step()
    assert len(scheduler.get_last_lr()) == 1


@pytest.mark.filterwarnings("ignore:torch.cuda.amp.GradScaler is enabled")  # No CUDA on the test runner.
@pytest.mark.parametrize(("dtype", "enabled"), [("float32", False), ("bfloat16", True), ("float16", True)])
def test_grad_scaler_follows_the_dtype(tiny_dinosr_config: DinoSRConfig, dtype: Dtype, *, enabled: bool) -> None:
    """Mixed precision keeps the scaler, which is what skips steps with non-finite gradients."""
    model = DinoSR(tiny_dinosr_config)
    cfg = OptimizerConfig(dtype=dtype)
    assert cfg.mixed_precision is enabled
    _, scaler, _ = build_optimizer(model, cfg)
    assert scaler.is_enabled() is (enabled and torch.cuda.is_available())
