# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
"""Optimization utilities."""

import math

import torch
from torch import GradScaler
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LinearLR, LRScheduler, SequentialLR

from spidr.config import OptimizerConfig


def build_scheduler(opt: Optimizer, cfg: OptimizerConfig) -> LRScheduler:
    init_lr_scale = cfg.init_lr_scale if cfg.warmup_steps > 0 else 1
    decay: LRScheduler
    if cfg.scheduler == "tristage":
        warmup = LinearLR(opt, start_factor=init_lr_scale, total_iters=cfg.warmup_steps)
        hold = LinearLR(opt, start_factor=1.0, total_iters=cfg.hold_steps)
        decay = LambdaLR(opt, lambda step: math.exp(math.log(cfg.final_lr_scale) * step / cfg.decay_steps))
        return SequentialLR(opt, [warmup, hold, decay], [cfg.warmup_steps, cfg.hold_steps + cfg.warmup_steps])
    if cfg.scheduler == "cosine":
        warmup = LinearLR(opt, start_factor=init_lr_scale, total_iters=cfg.warmup_steps)
        decay = CosineAnnealingLR(opt, cfg.max_steps - cfg.warmup_steps, cfg.final_lr_scale * cfg.lr)
        return SequentialLR(opt, [warmup, decay], [cfg.warmup_steps])
    if cfg.scheduler == "rsqrt":
        warmup = LinearLR(
            opt,
            start_factor=init_lr_scale,
            end_factor=1 / math.sqrt(1 + cfg.rsqrt_shift / cfg.rsqrt_timescale),
            total_iters=cfg.warmup_steps,
        )
        hold = LinearLR(opt, start_factor=1.0, total_iters=cfg.hold_steps)
        decay = LambdaLR(opt, lambda step: 1 / math.sqrt(1 + (step + cfg.rsqrt_shift) / cfg.rsqrt_timescale))
        return SequentialLR(opt, [warmup, hold, decay], [cfg.warmup_steps, cfg.hold_steps + cfg.warmup_steps])
    if cfg.scheduler == "constant":
        warmup = LinearLR(opt, start_factor=init_lr_scale, total_iters=cfg.warmup_steps)
        hold = LinearLR(opt, start_factor=1.0, total_iters=cfg.max_steps)
        return SequentialLR(opt, [warmup, hold], [cfg.warmup_steps])
    raise ValueError(f"Unknown scheduler: {cfg.scheduler}")


def build_optimizer(model: torch.nn.Module, cfg: OptimizerConfig) -> tuple[AdamW, GradScaler, LRScheduler]:
    if cfg.dtype not in {"float32", "float16", "bfloat16"}:
        raise ValueError(cfg.dtype)
    if cfg.dtype == "bfloat16" and torch.cuda.is_available() and torch.cuda.get_device_capability() < (8, 0):
        raise ValueError("Cannot use bfloat16 on this GPU (V100?), try again with float16")

    param_groups: list[dict[str, list[torch.nn.Parameter]]] = [{"params": []}, {"params": []}]
    for name, param in model.named_parameters():
        if any(name.startswith(exclude) for exclude in cfg.exclude_from_optimizer):
            continue
        group = 1 if any(name.startswith(freeze) for freeze in cfg.to_freeze) else 0
        param_groups[group]["params"].append(param)

    optimizer = AdamW(param_groups, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas, eps=cfg.eps, fused=True)
    scaler = GradScaler("cuda", enabled=cfg.mixed_precision)
    scheduler = build_scheduler(optimizer, cfg)
    return optimizer, scaler, scheduler
