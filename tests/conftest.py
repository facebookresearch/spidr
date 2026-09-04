# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
"""Shared fixtures: tiny model configurations and small audio corpora that run quickly on CPU."""

import tarfile
from pathlib import Path
from typing import TypedDict

import pytest
import torch
from torchcodec.encoders import AudioEncoder

from spidr.config import SAMPLE_RATE, DinoSRConfig, SpidRConfig


class _TinyModelKwargs(TypedDict):
    extractor_conv_layer_config: list[tuple[int, int, int]]
    encoder_embed_dim: int
    encoder_pos_conv_kernel: int
    encoder_pos_conv_groups: int
    encoder_pos_conv_depth: int
    encoder_num_layers: int
    encoder_num_heads: int
    encoder_ff_interm_features: int
    codebook_size: int
    num_codebooks: int


TINY_MODEL_KWARGS: _TinyModelKwargs = {
    "extractor_conv_layer_config": [(8, 10, 5), (8, 8, 4)],
    "encoder_embed_dim": 16,
    "encoder_pos_conv_kernel": 8,
    "encoder_pos_conv_groups": 2,
    "encoder_pos_conv_depth": 2,
    "encoder_num_layers": 2,
    "encoder_num_heads": 2,
    "encoder_ff_interm_features": 32,
    "codebook_size": 8,
    "num_codebooks": 2,
}

WAV_LENGTHS = [8_000, 12_000, 10_000]


@pytest.fixture
def tiny_dinosr_config() -> DinoSRConfig:
    return DinoSRConfig(**TINY_MODEL_KWARGS)


@pytest.fixture
def tiny_spidr_config() -> SpidRConfig:
    return SpidRConfig(**TINY_MODEL_KWARGS)


@pytest.fixture(scope="session")
def wav_lengths() -> list[int]:
    return list(WAV_LENGTHS)


@pytest.fixture(scope="session")
def wav_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A directory of mono 16 kHz wav files, encoded once for the whole session."""
    directory = tmp_path_factory.mktemp("wavs")
    generator = torch.Generator().manual_seed(0)
    for i, length in enumerate(WAV_LENGTHS):
        samples = torch.randn(1, length, generator=generator) * 0.1
        AudioEncoder(samples, sample_rate=SAMPLE_RATE).to_file(str(directory / f"sample{i}.wav"))
    return directory


@pytest.fixture(scope="session")
def wav_archive(wav_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An uncompressed tar archive holding the same wav files."""
    archive = tmp_path_factory.mktemp("archive") / "corpus.tar"
    with tarfile.open(archive, mode="w") as tar:
        for path in sorted(wav_dir.glob("*.wav")):
            tar.add(path, arcname=path.name)
    return archive
