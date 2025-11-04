# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
"""Speech dataset and batch sampler. Adapted from torchaudio."""

import abc
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor
from torch import distributed as dist
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import BatchSampler, DataLoader, Dataset, DistributedSampler
from torchcodec.decoders import AudioDecoder

from spidr.config import DEFAULT_CONV_LAYER_CONFIG, SAMPLE_RATE, DataConfig, MaskingConfig
from spidr.data.masks import MaskGenerator
from spidr.data.utils import bytes_from_archive, read_manifest


def verify_lengths(min_len: int, max_len: int, batch_size: int | None, max_token_count: int | None) -> None:
    exception = ""
    if not 0 <= min_len <= max_len:
        exception += "min_len must be less than or equal to max_len \n"
    if max_token_count is not None and batch_size is not None:
        exception += "max_token_count and batch_size cannot be set simultaneously \n"
    if max_token_count is None and batch_size is None:
        exception += "max_token_count or batch_size must be set \n"
    if max_token_count is not None and max_len > max_token_count:
        exception += "max_len must be less than or equal to max_token_count \n"
    if exception:
        raise ValueError(exception)


def get_buckets(lengths: list[int], num_buckets: int, uniform_limits: tuple[int, int] | None) -> dict[int, Tensor]:
    buckets: dict[int, list[int]] = {}
    if uniform_limits is not None:
        boundaries = torch.linspace(uniform_limits[0] - 1, uniform_limits[1] + 1, num_buckets + 1)
    else:
        boundaries = torch.quantile(
            torch.tensor(lengths, dtype=torch.float32),
            torch.linspace(0, 1, num_buckets + 1),
            interpolation="lower",
        )[1:]
    bucket_ids = torch.bucketize(torch.tensor(lengths), boundaries)
    for i in range(bucket_ids.size(0)):
        bucket_id = int(bucket_ids[i])
        if bucket_id in buckets:
            buckets[bucket_id].append(i)
        else:
            buckets[bucket_id] = [i]
    return dict(sorted([(k, torch.as_tensor(v, dtype=torch.int)) for k, v in buckets.items()]))


class BucketizeBatchSampler(BatchSampler):
    """Batch sampler that groups samples of similar length into buckets."""

    def __init__(
        self,
        *,
        lengths: list[int],
        num_buckets: int,
        min_len: int,
        max_len: int | None,
        max_token_count: int | None,
        batch_size: int | None,
        seed: int,
        bucket_method: Literal["uniform", "percentile"],
        shuffle: bool,
        drop_last: bool,
    ) -> None:
        if max_len is None:
            max_len = max(lengths)
        verify_lengths(min_len, max_len, batch_size, max_token_count)
        filtered_length_idx = [(min(length, max_len), i) for i, length in enumerate(lengths) if min_len <= length]
        if not filtered_length_idx:
            exception = "No samples with length in the range"
            raise ValueError(exception)

        sorted_filtered_length_idx = sorted(filtered_length_idx, key=lambda x: x[0])
        self.lengths = [e[0] for e in sorted_filtered_length_idx]
        self.indices = [e[1] for e in sorted_filtered_length_idx]
        self.max_token_count = max_token_count
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        if self.shuffle:
            self.generator = torch.Generator().manual_seed(self.seed)
        self.drop_last = drop_last
        if bucket_method == "uniform":
            uniform_limits = (min_len, max_len)
        elif bucket_method == "percentile":
            uniform_limits = None
        else:
            raise ValueError("bucket_method must be either 'uniform' or 'percentile'")
        self.buckets = get_buckets(self.lengths, num_buckets, uniform_limits)
        self._update_iter_list()

    def _update_iter_list(self) -> None:
        if self.shuffle:
            for k in self.buckets:
                new_idx = torch.randperm(self.buckets[k].size(0), generator=self.generator)
                self.buckets[k] = self.buckets[k][new_idx]
        self.iter_list = []
        total_len, batch = 0, []
        max_batch_size = self.max_token_count or self.batch_size
        for k in self.buckets:
            for i in range(self.buckets[k].size(0)):
                index = int(self.buckets[k][i])
                sample_length = self.lengths[index] if self.max_token_count else 1
                if total_len + sample_length <= max_batch_size:
                    batch.append(self.indices[index])
                    total_len += sample_length
                else:
                    self.iter_list.append(batch)
                    batch = [self.indices[index]]
                    total_len = sample_length
        if len(batch) > 0 and (self.max_token_count or not self.drop_last):
            self.iter_list.append(batch)

    def set_epoch(self, epoch: int) -> None:
        self.seed += epoch
        self.generator.manual_seed(self.seed)
        self._update_iter_list()

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self.iter_list)

    def __len__(self) -> int:
        return len(self.iter_list)


class DistributedBatchSampler(DistributedSampler):
    """Distributed sampler for BucketizeBatchSampler."""

    def __init__(self, batch_sampler: BucketizeBatchSampler, *, seed: int, shuffle: bool, drop_last: bool) -> None:
        self.batch_sampler = batch_sampler
        self.num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.shuffle = shuffle
        self.epoch = 0
        self.seed = seed
        self.drop_last = drop_last
        self.shuffle = shuffle
        indices = self.batch_sampler.iter_list
        if self.drop_last and len(indices) % self.num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when using this Sampler.
            self.num_samples = math.ceil((len(indices) - self.num_replicas) / self.num_replicas)
        else:
            self.num_samples = math.ceil(len(indices) / self.num_replicas)

    def set_epoch(self, epoch: int) -> None:
        self.batch_sampler.set_epoch(epoch)
        return super().set_epoch(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            perm = torch.randperm(len(self.batch_sampler.iter_list), generator=generator).tolist()
            indices = [self.batch_sampler.iter_list[i] for i in perm]
        else:
            indices = self.batch_sampler.iter_list
        if self.drop_last:
            self.total_size = len(indices) - len(indices) % self.num_replicas
        else:
            padding_size = self.num_replicas - len(indices) % self.num_replicas
            indices += indices[:padding_size]
            self.total_size = len(indices)
        self.num_samples = self.total_size // self.num_replicas
        self.subset = indices[self.rank : self.total_size : self.num_replicas]
        if len(self.subset) != self.num_samples:
            exception = f"Rank {self.rank} has subset of length {len(self.subset)} but expected {self.num_samples}"
            raise ValueError(exception)
        return iter(self.subset)

    def __len__(self) -> int:
        return self.num_samples


class SpeechDataset(Dataset, abc.ABC):
    """Dataset to load chunks of audio files."""

    def __init__(self, manifest_path: Path | str, *, normalize: bool) -> None:
        super().__init__()
        self.manifest = read_manifest(manifest_path)
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.manifest)

    @abc.abstractmethod
    def _load_audio(self, index: int) -> tuple[Tensor, int]:
        pass

    def __getitem__(self, index: int) -> Tensor:
        waveform, sr = self._load_audio(index)
        if sr != SAMPLE_RATE or waveform.shape[0] != 1:
            raise ValueError(index)
        if self.normalize:
            waveform = F.layer_norm(waveform, waveform.shape)
        return waveform.squeeze()


class SpeechDatasetFromArchive(SpeechDataset):
    def _load_audio(self, index: int) -> tuple[Tensor, int]:
        entry = self.manifest[index].to_dicts()[0]
        data = bytes_from_archive(entry["archive"], entry["byte_offset"], entry["byte_size"])
        samples = AudioDecoder(data).get_all_samples()
        return samples.data, samples.sample_rate


class SpeechDatasetFromFiles(SpeechDataset):
    def _load_audio(self, index: int) -> tuple[Tensor, int]:
        samples = AudioDecoder(self.manifest[index, "path"]).get_all_samples()
        return samples.data, samples.sample_rate


def speech_dataset(manifest_path: Path | str, *, normalize: bool) -> SpeechDataset:
    with Path(manifest_path).open("r", encoding="utf-8") as f:
        columns = set(f.readline().strip().split(","))
    if {"fileid", "path", "num_samples", "archive", "byte_offset", "byte_size"}.issubset(columns):
        return SpeechDatasetFromArchive(manifest_path, normalize=normalize)
    return SpeechDatasetFromFiles(manifest_path, normalize=normalize)


def conv_length(shapes: list[tuple[int, int, int]], length: Tensor) -> Tensor:
    for _, kernel_size, stride in shapes:
        length = torch.div(length - kernel_size, stride, rounding_mode="floor") + 1
        length = torch.max(torch.zeros_like(length), length)
    return length


def crop_audio(waveform: Tensor, num_samples: int, max_sample_size: int, *, rand_crop: bool) -> tuple[Tensor, int]:
    frame_offset = 0
    length = waveform.size(0)
    num_samples = min(num_samples, max_sample_size)
    if length > num_samples and rand_crop:
        frame_offset = int(torch.randint(length - num_samples, size=(1,)))
    elif length < num_samples:
        num_samples = length
    return waveform[frame_offset : frame_offset + num_samples], num_samples


class SpeechCollatorWithMasking:
    def __init__(
        self,
        mask_generator: MaskGenerator,
        *,
        max_sample_size: int,
        conv_layer_config: list[tuple[int, int, int]],
        enable_padding: bool,
        rand_crop: bool,
    ) -> None:
        self.mask_generator = mask_generator
        self.max_sample_size = max_sample_size
        self.conv_layer_config = conv_layer_config
        self.enable_padding = enable_padding
        self.rand_crop = rand_crop

    def __call__(self, batch: list[Tensor]) -> tuple[Tensor, Tensor, Tensor | None]:
        num_samples = max(len(wav) for wav in batch) if self.enable_padding else min(len(wav) for wav in batch)
        wavs_with_len = [crop_audio(wav, num_samples, self.max_sample_size, rand_crop=self.rand_crop) for wav in batch]
        wav_list, wav_lengths = zip(*wavs_with_len, strict=True)
        wavs = pad_sequence(wav_list, batch_first=True)
        lengths = conv_length(self.conv_layer_config, torch.tensor(wav_lengths))
        batch_size, max_len = wavs.size(0), int(lengths.max())
        padding_mask = torch.arange(max_len, device=lengths.device).expand(batch_size, max_len) >= lengths[:, None]
        attn_mask = ~padding_mask[:, None, None, :].expand(batch_size, 1, max_len, max_len)
        mask_indices = self.mask_generator(padding_mask)[0]
        return wavs, attn_mask, mask_indices


def build_dataloader(
    data_cfg: DataConfig,
    mask_cfg: MaskingConfig,
    *,
    conv_layer_config: list[tuple[int, int, int]] | None = None,
) -> DataLoader:
    dataset = speech_dataset(data_cfg.manifest, normalize=data_cfg.normalize)
    batch_sampler = DistributedBatchSampler(
        BucketizeBatchSampler(
            lengths=dataset.manifest["num_samples"].to_list(),
            num_buckets=data_cfg.num_buckets,
            min_len=data_cfg.min_sample_size,
            max_len=data_cfg.max_sample_size,
            max_token_count=data_cfg.max_batch_length,
            batch_size=None,
            seed=data_cfg.random_seed,
            bucket_method=data_cfg.bucket_method,
            shuffle=True,
            drop_last=data_cfg.drop_last,
        ),
        seed=data_cfg.random_seed,
        shuffle=True,
        drop_last=data_cfg.drop_last,
    )
    collate_fn = SpeechCollatorWithMasking(
        MaskGenerator(mask_cfg),
        max_sample_size=data_cfg.max_sample_size,
        conv_layer_config=conv_layer_config or DEFAULT_CONV_LAYER_CONFIG,
        enable_padding=data_cfg.enable_padding,
        rand_crop=data_cfg.rand_crop,
    )
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=data_cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=data_cfg.pin_memory,
        prefetch_factor=data_cfg.prefetch_factor,
        persistent_workers=data_cfg.persistent_workers,
        generator=torch.Generator().manual_seed(
            data_cfg.random_seed + (dist.get_rank() if dist.is_initialized() else 0)
        ),
    )
