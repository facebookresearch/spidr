# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
"""Environment setup."""

import logging
import os
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch._appdirs
import torch._inductor.config
from torch import distributed as dist

from spidr.tools import init_logger

logger = logging.getLogger()


@dataclass(frozen=True)
class DistributedEnvironment:
    global_rank: int
    local_rank: int
    world_size: int
    master_port: int
    master_addr: str


def distributed_environment() -> DistributedEnvironment:
    if os.getenv("LOCAL_RANK") is not None:  # torchrun job
        env = DistributedEnvironment(
            global_rank=int(os.environ["RANK"]),
            local_rank=int(os.environ["LOCAL_RANK"]),
            world_size=int(os.environ["WORLD_SIZE"]),
            master_port=int(os.environ["MASTER_PORT"]),
            master_addr=os.environ["MASTER_ADDR"],
        )
    elif os.getenv("SLURM_JOB_ID") is not None:  # Slurm job
        scontrol_cmd = ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]]
        master_addr = subprocess.check_output(scontrol_cmd, text=True).split()[0]
        master_port = random.Random(int(os.environ["SLURM_JOB_ID"])).randint(20_000, 60_000)
        env = DistributedEnvironment(
            global_rank=int(os.environ["SLURM_PROCID"]),
            local_rank=int(os.environ["SLURM_LOCALID"]),
            world_size=int(os.environ["SLURM_NTASKS"]),
            master_port=master_port,
            master_addr=master_addr,
        )
    else:
        env = DistributedEnvironment(  # Single GPU job
            global_rank=0,
            local_rank=0,
            world_size=1,
            master_port=random.Random(-1).randint(20_000, 60_000),
            master_addr="127.0.0.1",
        )
    os.environ["RANK"] = str(env.global_rank)
    os.environ["LOCAL_RANK"] = str(env.local_rank)
    os.environ["WORLD_SIZE"] = str(env.world_size)
    os.environ["MASTER_PORT"] = str(env.master_port)
    os.environ["MASTER_ADDR"] = env.master_addr
    return env


def setup_distributed() -> None:
    env = distributed_environment()
    logger.debug("World size %s, local rank %s, global rank %s", env.world_size, env.local_rank, env.global_rank)
    dist.init_process_group(backend="nccl")
    logger.debug("DDP initialized")


def setup_environment(**kwargs: str) -> None:
    if "SLURM_JOB_ID" in os.environ and (Path("/scratch") / os.environ["SLURM_JOB_ID"]).is_dir():
        cache = Path("/scratch") / os.environ["SLURM_JOB_ID"]
    else:
        cache = Path(torch._appdirs.user_cache_dir(appname="spidr"))
    cache.mkdir(exist_ok=True, parents=True)
    kwargs["TMPDIR"] = str(cache / "tmp")
    kwargs["TORCHINDUCTOR_CACHE_DIR"] = str(cache / "torchinductor")
    kwargs["TRITON_HOME"] = str(cache)

    for name, value in kwargs.items():
        if os.getenv(name) != str(value):
            os.environ[name] = str(value)
            logger.debug("Setting env: %s=%s", name, value)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def setup_pytorch(*, use_deterministic: bool) -> None:
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.backends.cudnn.benchmark = False  # Needed because of dynamic input size
    torch.autograd.set_detect_anomaly(mode=False, check_nan=True)
    if use_deterministic:
        torch.use_deterministic_algorithms(mode=True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch._inductor.config.fallback_random = True


def setup_training(seed: int, *, use_deterministic: bool = False) -> None:
    init_logger()
    setup_distributed()
    set_seed(seed + dist.get_rank())
    setup_pytorch(use_deterministic=use_deterministic)
    setup_environment()
