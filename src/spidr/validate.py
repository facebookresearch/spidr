# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
"""Run validation with an existing checkpoint."""

import json
import logging
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from spidr.config import Config, ResumeConfig
from spidr.data import build_dataloader
from spidr.environment import set_seed, setup_environment, setup_pytorch
from spidr.models import DinoSR, build_model
from spidr.tools import init_logger

logger = logging.getLogger()


@torch.no_grad()
def validate(model: DinoSR, loader: DataLoader, device: torch.device, dtype: torch.dtype) -> dict[str, float]:
    model.eval()
    total_loss, num_frames, num_batches = torch.zeros((), device=device), 0, len(loader)
    totals = defaultdict(lambda: torch.zeros((), device=device))
    for waveforms, attn_mask, mask in loader:
        with torch.autocast(device.type, dtype, enabled=dtype != torch.float32):
            loss, outputs = model(
                waveforms.to(device),
                mask_indices=mask.to(device),
                attention_mask=attn_mask.to(device) if attn_mask is not None else None,
            )
        num_frames += loss.numel()
        total_loss += loss.float().sum()
        for name, value in outputs.items():
            totals[name] += value.float()
    return {"loss": (total_loss / num_frames).item()} | {n: (v / num_batches).item() for n, v in totals.items()}


def validate_existing_checkpoint(cfg: Config, resume: ResumeConfig) -> None:
    logger.info("Starting validation for step %s", resume.step)
    init_logger()
    set_seed(cfg.run.random_seed)
    setup_pytorch(use_deterministic=cfg.run.use_deterministic)
    setup_environment()
    if resume.checkpoint is None:
        logger.error("No checkpoint found, validation failed")
        return
    logger.info("Loading model from %s", resume.checkpoint)
    model = build_model(model_type=cfg.run.model_type, checkpoint=resume.checkpoint).eval().to("cuda")
    dtype = getattr(torch, cfg.optimizer.dtype)
    for val_name, val_cfg in cfg.validation.items():
        logger.info("Validating on %s", val_name)
        loader = build_dataloader(val_cfg, cfg.masking, conv_layer_config=cfg.model.extractor_conv_layer_config)
        val_output = {"step": resume.step, "group": val_name} | validate(model, loader, torch.device("cuda"), dtype)
        logger.info("Validation on %s done: loss %s", val_name, val_output["loss"])
        with resume.results.open("a") as f:
            f.write(json.dumps(val_output) + "\n")
