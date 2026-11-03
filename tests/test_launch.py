# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
"""Tests for launching runs: Slurm options and distributed environment."""

import argparse
import os
import subprocess

import pytest

from spidr import environment
from spidr.environment import distributed_environment
from spidr.slurm import slurm_config_parse_args


def slurm_master_port(monkeypatch: pytest.MonkeyPatch, job_id: str, restart_count: str | None) -> int:
    env = {
        "SLURM_JOB_ID": job_id,
        "SLURM_JOB_NODELIST": "node[1-2]",
        "SLURM_PROCID": "0",
        "SLURM_LOCALID": "0",
        "SLURM_NTASKS": "2",
    }
    if restart_count is not None:
        env["SLURM_RESTART_COUNT"] = restart_count
    monkeypatch.setattr(os, "environ", env)
    monkeypatch.setattr(
        environment.subprocess,
        "run",
        lambda cmd, **_: subprocess.CompletedProcess(cmd, 0, stdout="node1\nnode2\n"),
    )
    env_info = distributed_environment()
    assert env_info.master_addr == "node1"
    return env_info.master_port


def test_master_port_changes_when_slurm_requeues_the_job(monkeypatch: pytest.MonkeyPatch) -> None:
    first = slurm_master_port(monkeypatch, "123456", None)
    assert slurm_master_port(monkeypatch, "123456", "0") == first
    assert slurm_master_port(monkeypatch, "123456", "1") != first
    assert 20_000 <= slurm_master_port(monkeypatch, "123456", "1") <= 32_000


def test_slurm_config_keeps_exclude() -> None:
    args = argparse.Namespace(account="acc", nodes=1, gpus_per_node=8, exclude="node[1-3]", qos=None)
    assert slurm_config_parse_args(args) == {"account": "acc", "nodes": 1, "gpus_per_node": 8, "exclude": "node[1-3]"}
