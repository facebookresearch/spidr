# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
"""Tests for checkpoint save/resume, including the feature extractor freeze transition."""

from pathlib import Path

import torch
from torch.optim.lr_scheduler import LRScheduler

from spidr.checkpoint import Checkpointer, remove_param_group
from spidr.config import DinoSRConfig, OptimizerConfig
from spidr.models import DinoSR
from spidr.optimizer import build_optimizer


def make_state(cfg: DinoSRConfig, seed: int) -> tuple[DinoSR, torch.optim.AdamW, torch.GradScaler, LRScheduler]:
    torch.manual_seed(seed)
    model = DinoSR(cfg)
    optimizer, scaler, scheduler = build_optimizer(model, OptimizerConfig(dtype="float32"))
    return model, optimizer, scaler, scheduler


def make_checkpointer(folder: Path, model, optimizer, scaler, scheduler) -> Checkpointer:
    checkpointer = Checkpointer(folder, interval=10)
    checkpointer.init_state(model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler)
    return checkpointer


def test_save_and_resume_round_trip(tiny_dinosr_config: DinoSRConfig, tmp_path: Path) -> None:
    model, optimizer, scaler, scheduler = make_state(tiny_dinosr_config, seed=0)
    checkpointer = make_checkpointer(tmp_path, model, optimizer, scaler, scheduler)
    assert not checkpointer.load_existing_run()
    assert not checkpointer.save(11, 1)  # Not on the save interval.
    assert checkpointer.save(10, 1)
    assert (tmp_path / "step_10.pt").is_file()

    other_model, other_optimizer, other_scaler, other_scheduler = make_state(tiny_dinosr_config, seed=1)
    other = make_checkpointer(tmp_path, other_model, other_optimizer, other_scaler, other_scheduler)
    assert other.load_existing_run()
    assert int(other.step) == 10
    assert int(other.epoch) == 1
    for name, param in model.state_dict().items():
        torch.testing.assert_close(param, other_model.state_dict()[name], msg=name)


def test_resume_after_freeze_step(tiny_dinosr_config: DinoSRConfig, tmp_path: Path) -> None:
    model, optimizer, scaler, scheduler = make_state(tiny_dinosr_config, seed=0)
    checkpointer = make_checkpointer(tmp_path, model, optimizer, scaler, scheduler)

    model.update_ema(model.freeze_step)  # Freezes the extractor and records the step.
    assert len(optimizer.param_groups) == 2
    remove_param_group(optimizer, scheduler, 1)
    assert len(optimizer.param_groups) == 1
    optimizer.step()  # The training loop always steps the optimizer first.
    scheduler.step()  # The scheduler must survive the param group surgery.
    assert checkpointer.save(model.freeze_step, 2, force=True)

    other_model, other_optimizer, other_scaler, other_scheduler = make_state(tiny_dinosr_config, seed=1)
    assert len(other_optimizer.param_groups) == 2
    other = make_checkpointer(tmp_path, other_model, other_optimizer, other_scaler, other_scheduler)
    assert other.load_existing_run()
    assert other_model._extractor_frozen
    assert all(not p.requires_grad for p in other_model.feature_extractor.parameters())
    assert len(other_optimizer.param_groups) == 1


def test_round_trip_preserves_codebook_buffers(tiny_dinosr_config: DinoSRConfig, tmp_path: Path) -> None:
    """The codebooks live in buffers, not parameters: they must survive save/load."""
    model, optimizer, scaler, scheduler = make_state(tiny_dinosr_config, seed=0)
    checkpointer = make_checkpointer(tmp_path, model, optimizer, scaler, scheduler)
    torch.manual_seed(0)
    model.train()(torch.randn(2, 1600), mask=torch.ones(2, 78, dtype=torch.bool))
    assert checkpointer.save(10, 1)

    other_model, *other_state = make_state(tiny_dinosr_config, seed=1)
    make_checkpointer(tmp_path, other_model, *other_state).load_existing_run()
    for expected, actual in zip(model.codebooks, other_model.codebooks, strict=True):
        torch.testing.assert_close(expected.codebook, actual.codebook)
        torch.testing.assert_close(expected.counts, actual.counts)


def test_keep_latest_purges_older_checkpoints(tiny_dinosr_config: DinoSRConfig, tmp_path: Path) -> None:
    model, optimizer, scaler, scheduler = make_state(tiny_dinosr_config, seed=0)
    checkpointer = Checkpointer(tmp_path, interval=10, keep_latest=2)
    checkpointer.init_state(model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler)
    for step in (10, 20, 30):
        assert checkpointer.save(step, 1)
    assert sorted(path.name for path in tmp_path.glob("step_*.pt")) == ["step_20.pt", "step_30.pt"]
    assert checkpointer.last == tmp_path / "step_30.pt"
