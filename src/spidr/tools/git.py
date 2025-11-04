# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
"""Retrieve and write git information."""

import importlib.resources
import subprocess
from pathlib import Path
from typing import NamedTuple


class GitInfo(NamedTuple):
    branch: str
    commit: str
    diff: str


def git_available() -> bool:
    git_cmd = ["git", "rev-parse", "--is-inside-work-tree"]
    try:
        return subprocess.run(git_cmd, capture_output=True, check=False, text=True).returncode == 0
    except FileNotFoundError:
        return False


def git_info() -> GitInfo:
    root = Path(importlib.resources.files(__package__)).parent.parent
    git_cmd = ["git", "-C", root, "rev-parse", "HEAD", "--abbrev-ref", "HEAD"]
    info = subprocess.run(git_cmd, capture_output=True, check=True, text=True)
    commit, branch = info.stdout.strip().split("\n")
    git_diff_cmd = ["git", "-C", root, "diff"]
    diff = subprocess.run(git_diff_cmd, capture_output=True, check=True, text=True)
    return GitInfo(branch=branch, commit=commit, diff=diff.stdout)


def write_git_info_if_available(path: str | Path) -> None:
    if not git_available():
        return
    info = git_info()
    (Path(path) / "git-info.txt").write_text(f"branch: {info.branch}\ncommit: {info.commit}\n")
    (Path(path) / "git-diff.patch").write_text(info.diff)
