# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
"""Mask creation."""

import torch
from torch import Tensor, nn

from spidr.config import MaskingConfig, MaskingStrategy


def mask_lengths(
    num_mask: int, mask_length: int, mask_other: float, mask_type: MaskingStrategy, generator: torch.Generator | None
) -> torch.Tensor:
    match mask_type:
        case "static":
            return torch.full((num_mask,), mask_length)
        case "uniform":
            return torch.randint(int(mask_other), mask_length * 2 + 1, size=(num_mask,), generator=generator)
        case "normal":
            lengths = torch.normal(mask_length, mask_other, size=(num_mask,), generator=generator)
            return torch.maximum(torch.ones(1), torch.round(lengths)).int()
        case "poisson":
            lengths = torch.poisson(torch.rand(num_mask) * mask_length, generator=generator)
            return torch.round(lengths).int()
    raise ValueError(f"unknown mask strategy: {mask_type}")


def mask_id_overlap(lengths: torch.Tensor, num_mask: int, sz: int, generator: torch.Generator | None) -> torch.Tensor:
    min_len = min(lengths)
    if sz - min_len <= num_mask:
        min_len = sz - num_mask - 1
    mask_idc = torch.randperm(sz - min_len, generator=generator)[:num_mask]
    return torch.tensor([mask_idc[j] + offset for j in range(len(mask_idc)) for offset in range(lengths[j])])


def mask_id_no_overlap(
    lengths: torch.Tensor, min_space: int, sz: int, generator: torch.Generator | None
) -> torch.Tensor:
    mask_idc_list: list[int] = []

    def arrange(mask_idc_list: list[int], s: int, e: int, length: int, keep_length: int) -> list[tuple[int, int]]:
        span_start = int(torch.randint(s, e - length, size=(1,), generator=generator))
        mask_idc_list.extend(span_start + i for i in range(length))

        new_parts = []
        if span_start - s - min_space >= keep_length:
            new_parts.append((s, span_start - min_space + 1))
        if e - span_start - keep_length - min_space > keep_length:
            new_parts.append((span_start + length + min_space, e))
        return new_parts

    parts = [(0, sz)]
    min_length = min(lengths)
    for length in sorted(lengths, reverse=True):
        lens = torch.tensor([e - s for s, e in parts], dtype=torch.int)
        lens[lens < length + min_space] = 0
        l_sum = lens.sum()
        if l_sum == 0:
            break
        probs = lens / l_sum
        c = torch.distributions.Categorical(probs).sample()  # WARNING: does not accept `generator`
        s, e = parts.pop(c)
        parts.extend(arrange(mask_idc_list, s, e, length, min_length))
    return torch.tensor(mask_idc_list)


def compute_mask_indices(
    shape: tuple[int, int],
    padding_mask: Tensor | None,
    mask_prob: float,
    mask_length: int,
    mask_type: MaskingStrategy,
    mask_other: float,
    min_masks: int,
    min_space: int,
    *,
    no_overlap: bool,
    generator: torch.Generator | None,
) -> Tensor:
    """Compute random mask spans for a given shape."""
    batch_size, frame = shape
    mask = torch.full((batch_size, frame), fill_value=False)
    # add a random number for probabilistic rounding
    all_num_mask = int(mask_prob * frame / float(mask_length) + torch.rand(1, generator=generator))
    all_num_mask = max(min_masks, all_num_mask)

    mask_idcs = []
    for i in range(batch_size):
        if padding_mask is not None:
            sz = frame - padding_mask[i].long().sum().item()
            # add a random number for probabilistic rounding
            num_mask = int(mask_prob * sz / float(mask_length) + torch.rand(1, generator=generator))
            num_mask = max(min_masks, num_mask)
        else:
            sz, num_mask = frame, all_num_mask
        lengths = mask_lengths(num_mask, mask_length, mask_other, mask_type, generator)
        if sum(lengths) == 0:
            lengths[0] = min(mask_length, sz - 1)
        mask_idc = (
            mask_id_no_overlap(lengths, min_space, sz, generator)
            if no_overlap
            else mask_id_overlap(lengths, num_mask, sz, generator)
        )
        mask_idcs.append(torch.unique(mask_idc[mask_idc < sz]))

    min_len = min(len(m) for m in mask_idcs)
    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > min_len:
            idx = torch.randperm(len(mask_idc), generator=generator)[:min_len].long()
            mask[i, mask_idc[idx]] = True
        else:
            mask[i, mask_idc] = True
    return mask


class MaskGenerator(nn.Module):
    """Generate the masks for masked prediction."""

    def __init__(self, config: MaskingConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        padding_mask: Tensor,
        *,
        channels: int | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor | None, Tensor | None]:
        batch, time = padding_mask.shape
        if self.config.mask_prob > 0:
            mask_indices = compute_mask_indices(
                (batch, time),
                padding_mask,
                self.config.mask_prob,
                self.config.mask_length,
                self.config.mask_selection,
                self.config.mask_other,
                min_masks=2,
                min_space=self.config.mask_min_space,
                no_overlap=self.config.no_mask_overlap,
                generator=generator,
            )
        else:
            mask_indices = None
        if self.config.mask_channel_prob > 0:
            if channels is None:
                raise ValueError("Must set 'channels' to mask channel-wise")
            mask_channel_indices = compute_mask_indices(
                (batch, channels),
                None,
                self.config.mask_channel_prob,
                self.config.mask_channel_length,
                self.config.mask_channel_selection,
                self.config.mask_channel_other,
                min_masks=0,
                min_space=self.config.mask_channel_min_space,
                no_overlap=self.config.no_mask_channel_overlap,
                generator=generator,
            )
            mask_channel_indices = mask_channel_indices.unsqueeze(1).expand(-1, time, -1)
        else:
            mask_channel_indices = None
        return mask_indices, mask_channel_indices
