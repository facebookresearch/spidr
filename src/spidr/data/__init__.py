# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
"""Data loading and dataset utilities."""

from spidr.data.dataset import build_dataloader, speech_dataset
from spidr.data.utils import num_samples, read_manifest

__all__ = ["build_dataloader", "num_samples", "read_manifest", "speech_dataset"]
