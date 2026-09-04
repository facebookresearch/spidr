# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
"""Tests for the batch samplers, the collator, and mask generation."""

import math
import random

import pytest
import torch

from spidr.config import MaskingConfig
from spidr.data.dataset import (
    BucketizeBatchSampler,
    DistributedBatchSampler,
    SpeechCollatorWithMasking,
    conv_length,
    verify_lengths,
)
from spidr.data.masks import MaskGenerator

CONV_LAYER_CONFIG = [(8, 10, 5), (8, 8, 4)]
NUM_LENGTHS = 200
MIN_LEN, MAX_LEN = 2_000, 20_000
MAX_TOKEN_COUNT = 40_000


def make_lengths() -> list[int]:
    rng = random.Random(0)
    return [rng.randint(MIN_LEN, MAX_LEN) for _ in range(NUM_LENGTHS)]


def make_bucket_sampler(seed: int = 7) -> BucketizeBatchSampler:
    return BucketizeBatchSampler(
        lengths=make_lengths(),
        num_buckets=10,
        min_len=MIN_LEN,
        max_len=MAX_LEN,
        max_token_count=MAX_TOKEN_COUNT,
        batch_size=None,
        seed=seed,
        bucket_method="uniform",
        shuffle=True,
        drop_last=False,
    )


def test_bucket_sampler_covers_all_samples() -> None:
    sampler = make_bucket_sampler()
    indices = [index for batch in sampler for index in batch]
    assert sorted(indices) == list(range(NUM_LENGTHS))


def test_bucket_sampler_respects_token_budget() -> None:
    lengths = make_lengths()
    for batch in make_bucket_sampler():
        assert sum(min(lengths[index], MAX_LEN) for index in batch) <= MAX_TOKEN_COUNT


def test_set_epoch_is_history_independent() -> None:
    resumed = make_bucket_sampler()
    resumed.set_epoch(3)
    uninterrupted = make_bucket_sampler()
    for epoch in (1, 2, 3):
        uninterrupted.set_epoch(epoch)
    assert list(resumed) == list(uninterrupted)


def test_distributed_sampler_single_replica_has_no_padding() -> None:
    bucket_sampler = make_bucket_sampler()
    sampler = DistributedBatchSampler(bucket_sampler, seed=0, shuffle=False, drop_last=False)
    assert list(sampler) == list(bucket_sampler)


def test_distributed_sampler_partitions_all_batches() -> None:
    bucket_sampler = make_bucket_sampler()
    sampler = DistributedBatchSampler(bucket_sampler, seed=0, shuffle=False, drop_last=False)
    sampler.num_replicas = 4
    per_rank = []
    for rank in range(4):
        sampler.rank = rank
        per_rank.append(list(sampler))
    assert len({len(subset) for subset in per_rank}) == 1  # Same number of batches on every rank.
    seen = [tuple(batch) for subset in per_rank for batch in subset]
    expected = [tuple(batch) for batch in bucket_sampler]
    assert set(seen) == set(expected)
    assert len(seen) - len(expected) < 4  # At most num_replicas - 1 padding batches.


def make_collator(*, enable_padding: bool) -> SpeechCollatorWithMasking:
    return SpeechCollatorWithMasking(
        MaskGenerator(MaskingConfig()),
        max_sample_size=320_000,
        conv_layer_config=CONV_LAYER_CONFIG,
        enable_padding=enable_padding,
        rand_crop=True,
    )


def test_collator_without_padding_returns_no_attention_mask() -> None:
    torch.manual_seed(0)
    batch = [torch.randn(4_000), torch.randn(5_000), torch.randn(4_500)]
    wavs, attn_mask, mask_indices = make_collator(enable_padding=False)(batch)
    assert wavs.shape == (3, 4_000)  # Cropped to the shortest waveform: no padding.
    assert attn_mask is None
    assert mask_indices is not None
    num_frames = int(conv_length(CONV_LAYER_CONFIG, torch.tensor([4_000]))[0])
    assert mask_indices.shape == (3, num_frames)


def test_collator_with_padding_returns_broadcastable_mask() -> None:
    torch.manual_seed(0)
    batch = [torch.randn(4_000), torch.randn(5_000), torch.randn(4_500)]
    wavs, attn_mask, mask_indices = make_collator(enable_padding=True)(batch)
    lengths = conv_length(CONV_LAYER_CONFIG, torch.tensor([4_000, 5_000, 4_500]))
    max_frames = int(lengths.max())
    assert wavs.shape == (3, 5_000)
    assert attn_mask is not None
    assert mask_indices is not None
    assert attn_mask.shape == (3, 1, 1, max_frames)
    assert attn_mask.dtype == torch.bool
    for i, length in enumerate(lengths.tolist()):
        assert attn_mask[i, 0, 0, :length].all()
        assert not attn_mask[i, 0, 0, length:].any()
    assert mask_indices.shape == (3, max_frames)
    assert not (mask_indices & ~attn_mask[:, 0, 0, :]).any()  # No masking inside padding.


def test_mask_generation_is_deterministic_with_generator() -> None:
    padding_mask = torch.zeros(2, 50, dtype=torch.bool)
    for no_mask_overlap in (False, True):
        generate = MaskGenerator(MaskingConfig(no_mask_overlap=no_mask_overlap))
        first = generate(padding_mask, generator=torch.Generator().manual_seed(0))[0]
        second = generate(padding_mask, generator=torch.Generator().manual_seed(0))[0]
        assert torch.equal(first, second)


def test_distributed_sampler_does_not_mutate_the_wrapped_sampler() -> None:
    """`__iter__` pads a copy: the wrapped sampler's batch list must survive repeated iteration."""
    bucket_sampler = make_bucket_sampler()
    sampler = DistributedBatchSampler(bucket_sampler, seed=0, shuffle=False, drop_last=False)
    sampler.num_replicas = 3
    before = len(bucket_sampler.iter_list)
    first, second = list(sampler), list(sampler)
    assert len(bucket_sampler.iter_list) == before
    assert first == second


def test_distributed_sampler_len_matches_what_it_yields() -> None:
    """`validate` divides its metrics by len(loader), so the two must agree before iterating."""
    bucket_sampler = make_bucket_sampler()
    sampler = DistributedBatchSampler(bucket_sampler, seed=0, shuffle=True, drop_last=False)
    assert len(sampler) == len(list(sampler))  # __init__ and __iter__ must agree on the count.
    for num_replicas in (3, 4):
        sampler.num_replicas = num_replicas
        yielded = len(list(sampler))
        assert len(sampler) == yielded
        assert yielded == math.ceil(len(bucket_sampler.iter_list) / num_replicas)


def test_bucket_sampler_set_epoch_without_shuffling() -> None:
    """The generator only exists when shuffling; set_epoch must not reach for it otherwise."""
    sampler = BucketizeBatchSampler(
        lengths=make_lengths(),
        num_buckets=10,
        min_len=MIN_LEN,
        max_len=MAX_LEN,
        max_token_count=MAX_TOKEN_COUNT,
        batch_size=None,
        seed=7,
        bucket_method="uniform",
        shuffle=False,
        drop_last=False,
    )
    before = list(sampler)
    sampler.set_epoch(1)
    assert list(sampler) == before


def test_bucket_sampler_batch_size_mode() -> None:
    sampler = BucketizeBatchSampler(
        lengths=make_lengths(),
        num_buckets=10,
        min_len=MIN_LEN,
        max_len=MAX_LEN,
        max_token_count=None,
        batch_size=8,
        seed=7,
        bucket_method="uniform",
        shuffle=False,
        drop_last=False,
    )
    batches = list(sampler)
    assert all(len(batch) <= 8 for batch in batches)
    assert sorted(index for batch in batches for index in batch) == list(range(NUM_LENGTHS))


def test_verify_lengths_rejects_ambiguous_batching() -> None:
    with pytest.raises(ValueError, match="max_token_count or batch_size must be set"):
        verify_lengths(0, 10, None, None)
    with pytest.raises(ValueError, match="cannot be set simultaneously"):
        verify_lengths(0, 10, 8, 100)
