# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
"""Launch Slurm jobs with submitit."""

import argparse
import importlib.resources
import json
import logging
import os
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import ParamSpec, TypeVar

import submitit
from submitit.core.utils import FailedJobError

from spidr.config import SlurmConfig
from spidr.tools import write_git_info_if_available

logger = logging.getLogger()


def current_codebase_path() -> Path:
    import spidr  # noqa: PLC0415

    return Path(spidr.__file__).parent.parent


def copy_code_to_submitit_folder(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(importlib.resources.files(__package__)).parent.parent
    logger.info("Copying: %s to: %s", root, output_dir)
    rsync_cmd = (
        f"rsync -ar --copy-links "
        f"--exclude='.git/' "
        f"--exclude './data/' "
        f"--exclude './paper/' "
        f"--exclude __pycache__ "
        f"--exclude '*.pyc' "
        f"--exclude '*.ipynb' "
        f"--exclude '*.err' "
        f"--exclude '*.out' "
        f"--exclude '*.log' "
        f"--exclude '*.pt' "
        f"--exclude '.ruff_cache/' "
        f"--exclude '.mypy_cache/' "
        f"{root}/ {output_dir}"
    )
    subprocess.call([rsync_cmd], shell=True)
    logger.info("Copy done.")


T = TypeVar("T")
P = ParamSpec("P")


def make_checkpointable(fn: Callable[P, T]) -> Callable[P, T]:  # noqa: UP047
    # Disable linting because "Only parameter specification variables defined in global scope can be pickled."
    def checkpoint(*args: P.args, **kwargs: P.kwargs) -> submitit.helpers.DelayedSubmission:
        return submitit.helpers.DelayedSubmission(fn, args, kwargs)

    fn.checkpoint = checkpoint  # ty: ignore[unresolved-attribute]
    return fn


def default_train_job_config(cluster_name: str | None) -> dict:
    match cluster_name:
        case None:
            return {}
        case "JZ_V100":
            return {"time": 1200, "cpus_per_task": 10, "mem_per_gpu": None, "constraint": None}
        case "JZ_A100":
            return {"time": 1200, "cpus_per_task": 8, "mem_per_gpu": None, "constraint": "a100"}
        case "JZ_H100":
            return {"time": 1200, "cpus_per_task": 24, "mem_per_gpu": None, "constraint": "h100"}
        case "FAIR_AWS_A100":
            return {"time": 4320, "cpus_per_task": 12, "mem_per_gpu": "80g", "constraint": None}
    raise ValueError(f"Unknown cluster {cluster_name}")


def validation_job_config() -> SlurmConfig:
    try:
        account, cpus_per_task = os.environ["SLURM_JOB_ACCOUNT"], os.environ["SLURM_CPUS_PER_TASK"]
    except KeyError as error:
        raise ValueError("This function should only be used inside a SLURM job.") from error
    return SlurmConfig(account=account, nodes=1, gpus_per_node=1, time=60, cpus_per_task=int(cpus_per_task))


def slurm_config_parse_args(args: argparse.Namespace) -> SlurmConfig:
    config = SlurmConfig(account=args.account, nodes=args.nodes, gpus_per_node=args.gpus_per_node)
    for key in SlurmConfig.__optional_keys__:
        if hasattr(args, key) and getattr(args, key) is not None:
            config[key] = getattr(args, key)  # ty: ignore[invalid-key]
    return config


@submitit.helpers.clean_env()
def launch_with_submitit(
    job_name: str,
    jobs: list[tuple[Callable, tuple]],
    dump: str | Path,
    slurm: SlurmConfig,
    *,
    copy_code: bool = True,
) -> bool:
    launch_time = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    folder = Path(dump) / job_name
    if folder.is_dir():
        folder = Path(dump) / f"{job_name}_{launch_time}"
    folder.mkdir(parents=True)

    if copy_code:
        copy_code_to_submitit_folder(folder / "code")
        codebase_path = folder / "code" / "src"
    else:  # No-op in most situations, but correctly handles nested cases
        codebase_path = current_codebase_path()

    write_git_info_if_available(folder)
    job_info = {"job_name": job_name, "launch_time": launch_time} | slurm
    (folder / "job-info.json").write_text(json.dumps(job_info, indent=2))

    os.chdir(codebase_path)
    logger.info("Launching from %s", codebase_path)
    executor = submitit.SlurmExecutor(folder=folder / "%j")
    executor.update_parameters(job_name=job_name, ntasks_per_node=slurm["gpus_per_node"], **slurm)
    try:
        with executor.batch():
            for fn, args in jobs:
                executor.submit(make_checkpointable(fn), *args)
    except FailedJobError as error:
        logger.warning("Didn't managed to submit %s: %s", job_name, error)
        return False
    logger.info("Submitted %s (%s jobs)", job_name, len(jobs))
    return True
