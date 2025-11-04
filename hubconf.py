# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
"""torch.hub configuration."""

from pathlib import Path

from torch.hub import _add_to_sys_path

dependencies = ["torch", "numpy"]

with _add_to_sys_path(str(Path(__file__).parent / "src")):
    from spidr.models import dinosr_base_original, dinosr_base_reproduced, spidr_base  # noqa: F401
