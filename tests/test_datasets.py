# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
"""Tests that decode real audio: manifests, the archive mmap cache, and the dataset classes."""

from pathlib import Path

import pytest
import torch
from torchcodec.encoders import AudioEncoder

from spidr.config import SAMPLE_RATE, DataConfig, MaskingConfig
from spidr.data.dataset import SpeechDatasetFromArchive, SpeechDatasetFromFiles, build_dataloader, speech_dataset
from spidr.data.utils import _archive_mmap, bytes_from_archive, num_samples, read_manifest
from spidr.data.write_manifest import write_manifest, write_manifest_tar


@pytest.fixture
def file_manifest(wav_dir: Path, tmp_path: Path) -> Path:
    manifest = tmp_path / "files.csv"
    write_manifest(wav_dir, manifest)
    return manifest


@pytest.fixture
def archive_manifest(wav_archive: Path, tmp_path: Path) -> Path:
    manifest = tmp_path / "archive.csv"
    write_manifest_tar(str(wav_archive), manifest)
    return manifest


def test_num_samples_reads_the_header(wav_dir: Path, wav_lengths: list[int]) -> None:
    for path in sorted(wav_dir.glob("*.wav")):
        assert num_samples(path, verify=True) in wav_lengths


def test_num_samples_rejects_undecodable_input(tmp_path: Path) -> None:
    path = tmp_path / "not-audio.wav"
    path.write_bytes(b"definitely not a wav file")
    with pytest.raises(RuntimeError, match="Could not open input file"):
        num_samples(path)


def test_write_manifest_lists_every_file(file_manifest: Path, wav_dir: Path, wav_lengths: list[int]) -> None:
    manifest = read_manifest(file_manifest)
    assert len(manifest) == len(wav_lengths)
    assert set(manifest["num_samples"].to_list()) == set(wav_lengths)
    assert {Path(p).parent for p in manifest["path"].to_list()} == {wav_dir}


def test_speech_dataset_dispatches_on_manifest_columns(file_manifest: Path, archive_manifest: Path) -> None:
    assert isinstance(speech_dataset(file_manifest, normalize=False), SpeechDatasetFromFiles)
    assert isinstance(speech_dataset(archive_manifest, normalize=False), SpeechDatasetFromArchive)


@pytest.mark.parametrize("normalize", [False, True])
def test_dataset_from_files_decodes_mono_audio(
    file_manifest: Path, wav_lengths: list[int], *, normalize: bool
) -> None:
    dataset = speech_dataset(file_manifest, normalize=normalize)
    assert len(dataset) == len(wav_lengths)
    for index in range(len(dataset)):
        waveform = dataset[index]
        assert waveform.ndim == 1
        assert waveform.shape[0] in wav_lengths
        assert torch.isfinite(waveform).all()
        if normalize:
            torch.testing.assert_close(waveform.mean(), torch.zeros(()), atol=1e-5, rtol=0)
            torch.testing.assert_close(waveform.std(unbiased=False), torch.ones(()), atol=1e-3, rtol=0)


def test_archive_and_file_datasets_decode_identically(file_manifest: Path, archive_manifest: Path) -> None:
    from_files = speech_dataset(file_manifest, normalize=False)
    from_archive = speech_dataset(archive_manifest, normalize=False)
    by_length = {int(from_files[i].shape[0]): from_files[i] for i in range(len(from_files))}
    for index in range(len(from_archive)):
        waveform = from_archive[index]
        torch.testing.assert_close(waveform, by_length[int(waveform.shape[0])])


def test_dataset_rejects_wrong_sample_rate(tmp_path: Path) -> None:
    other = tmp_path / "other"
    other.mkdir()
    AudioEncoder(torch.zeros(1, 4_000), sample_rate=SAMPLE_RATE // 2).to_file(str(other / "slow.wav"))
    manifest = tmp_path / "slow.csv"
    write_manifest(other, manifest)
    dataset = speech_dataset(manifest, normalize=False)
    with pytest.raises(ValueError, match="expected mono audio"):
        _ = dataset[0]


def test_archive_mmap_is_cached_per_path(wav_archive: Path, archive_manifest: Path) -> None:
    manifest = read_manifest(archive_manifest)
    first = bytes_from_archive(wav_archive, manifest[0, "byte_offset"], manifest[0, "byte_size"])
    second = bytes_from_archive(wav_archive, manifest[0, "byte_offset"], manifest[0, "byte_size"])
    assert first == second
    assert _archive_mmap.cache_info().hits >= 1


def test_build_dataloader_yields_collated_batches(file_manifest: Path) -> None:
    loader = build_dataloader(
        DataConfig(
            manifest=str(file_manifest),
            num_workers=1,
            persistent_workers=False,
            pin_memory=False,
            min_sample_size=1_000,
        ),
        MaskingConfig(),
        conv_layer_config=[(8, 10, 5), (8, 8, 4)],
    )
    waveforms, attn_mask, mask = next(iter(loader))
    assert waveforms.ndim == 2
    assert attn_mask is None  # enable_padding is off by default, so batches are cropped.
    assert mask.shape[0] == waveforms.shape[0]
    assert mask.dtype == torch.int64  # Masked positions, not a boolean mask.
