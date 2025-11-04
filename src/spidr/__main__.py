# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
"""Entry point for launching training runs."""

import argparse
import os
from pathlib import Path

from spidr.config import read_config
from spidr.slurm import default_train_job_config, launch_with_submitit, slurm_config_parse_args
from spidr.tools import init_logger
from spidr.train import train

if __name__ == "__main__":
    cluster = default_train_job_config(os.getenv("SPIDR_CLUSTER"))
    parser = argparse.ArgumentParser(
        description="Launch training runs.",
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("configs", type=Path, nargs="+", help="TOML config file(s) for the training run(s).")
    parser.add_argument("-A", "--account", type=str, required=True, help="Slurm account")
    parser.add_argument("-N", "--nodes", type=int, required=True, help="Number of nodes")
    parser.add_argument("-G", "--gpus-per-node", type=int, required=True, help="GPUs per node")
    parser.add_argument("-c", "--cpus-per-task", type=int, help="CPUs per task", default=cluster.get("cpus_per_task"))
    parser.add_argument("--mem-per-gpu", type=str, help="Memory per GPU", default=cluster.get("mem_per_gpu"))
    parser.add_argument("-t", "--time", type=int, help="Time limit in minutes", default=cluster.get("time"))
    parser.add_argument("-C", "--constraint", type=str, help="Slurm constraint", default=cluster.get("constraint"))
    parser.add_argument("-p", "--partition", type=str, help="Slurm partition", default=cluster.get("partition"))
    parser.add_argument("-q", "--qos", type=str, help="Slurm QoS", default=cluster.get("qos"))
    parser.add_argument("--dump", type=Path, help="Submitit dump", default=os.getenv("SPIDR_SUBMITIT_FOLDER"))
    args = parser.parse_args()

    if args.dump is None:
        parser.error("Dump directory must be specified with --dump or by setting $SPIDR_SUBMITIT_FOLDER")
    init_logger()
    jobs = [(train, (read_config(cfg),)) for cfg in args.configs]
    name = jobs[0][1][0].run.wandb_name
    launch_with_submitit(name, jobs, args.dump, slurm_config_parse_args(args))
