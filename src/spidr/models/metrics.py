# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
"""Metrics."""

import itertools
import math
from collections.abc import Iterable, Sequence
from itertools import groupby
from pathlib import Path
from typing import NamedTuple

import torch
import torch.compiler
from torch import Tensor
from torch import distributed as dist


@torch.no_grad()
@torch.compiler.disable
def perplexity(y: Tensor, *, tokenwise: bool = False) -> Tensor:
    if tokenwise:
        y = torch.exp2((-y * (y + 1e-8).log2()).sum(-1))
    if dist.is_initialized():
        n = torch.tensor([y.size(0)], device=y.device)
        y = y.sum(0)
        dist.all_reduce(n)
        dist.all_reduce(y)
        y /= n
    else:
        y = y.mean(0)
    if not tokenwise:
        y = torch.exp2((-y * (y + 1e-8).log2()).sum())
    return y


class NonFiniteError(Exception):
    """Non finite values were found."""


def params_norm(
    params: torch.Tensor | Iterable[torch.Tensor], norm_type: float = 2.0, *, error_if_nonfinite: bool = False
) -> torch.Tensor:
    params = (params,) if isinstance(params, torch.Tensor) else tuple(params)
    total_norm = torch.linalg.norm(torch.stack(torch._foreach_norm(params, norm_type)), norm_type)
    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise NonFiniteError
    return total_norm


PhonesAndUnits = tuple[list[str], list[int]]


def build_count_matrix(
    data: dict[str | Path, PhonesAndUnits],
    num_units: int,
    num_phones: int,
    repeat: int,
    max_length_diff: int = 2,
) -> tuple[torch.Tensor, list[str]]:
    phone_to_index, idx, phone_indices, unit_indices = {}, 0, [], []
    for path, phones_and_units in data.items():
        phones = phones_and_units[0]
        units = list(itertools.chain.from_iterable(itertools.repeat(unit, repeat) for unit in phones_and_units[1]))
        if abs(len(phones) - len(units)) > max_length_diff:
            raise ValueError(
                f"More than {max_length_diff} frames of difference "
                f" after repetition ({len(phones)} vs {len(units)}) in {path}"
            )
        for phone, unit in zip(phones, units, strict=False):
            if phone not in phone_to_index:
                phone_to_index[phone] = idx
                idx += 1
            if phone_to_index[phone] >= num_phones or unit >= num_units:
                raise IndexError(
                    f"Phone '{phone}' or code '{unit}' is out of bounds. "
                    f"Adjust num_phones ({num_phones}) or num_units ({num_units}) "
                )
            phone_indices.append(phone_to_index[phone])
            unit_indices.append(unit)

    flattened_indices = torch.tensor(phone_indices) * num_units + torch.tensor(unit_indices)
    count_flat = torch.bincount(flattened_indices, minlength=num_phones * num_units)
    count = count_flat.reshape(num_phones, num_units)
    most_frequent_phones = torch.argsort(count.sum(dim=1), descending=True)
    phone_order = [{v: k for k, v in phone_to_index.items()}[idx.item()] for idx in most_frequent_phones]
    count = count[most_frequent_phones]
    return count, phone_order


class UnitsQuality(NamedTuple):
    phone_purity: float
    cluster_purity: float
    pnmi: float
    codebook_perplexity: float
    active_codewords: int
    bitrate: float


def bitrate(*, num_units: int, sequences: Iterable[Sequence[int]] | None = None, frame_step: int = 20) -> float:
    """Bitrate in bits per second.

    :param num_units: Number of distinct units. Equivalent to the codebook size.
    :param sequences: Sequences of units.
        If provided, deduplicate consecutive units in each sequence.
        If None, assume that one unit is produced every `frame_step` ms.
    :param frame_step: Frame step in milliseconds.
    """
    if sequences is None:
        return math.log2(num_units) * 1000 / frame_step
    total_deduplicated_units, total_seconds = 0, 0
    for units in sequences:
        total_seconds += frame_step * len(units) / 1000  # frame_step is in ms
        total_deduplicated_units += len(list(groupby(units)))
    num_units_per_second = total_deduplicated_units / total_seconds
    return math.log2(num_units) * num_units_per_second


def units_quality(
    data: dict[str | Path, PhonesAndUnits],
    *,
    num_units: int,
    num_phones: int,
    frame_step: int = 20,
    alignment_step: int = 10,
    eps: float = 1e-10,
) -> UnitsQuality:
    count = build_count_matrix(data, num_units, num_phones, frame_step // alignment_step)[0]
    proba = count / count.sum()
    phone_purity = proba.max(dim=0).values.sum()
    cluster_purity = proba.max(dim=1).values.sum()
    px = proba.sum(dim=1, keepdim=True)
    py = proba.sum(dim=0, keepdim=True)
    mutual_info = (proba * torch.log(proba / (px @ py + eps) + eps)).sum()
    entropy_x = (-px * torch.log(px + eps)).sum()
    pnmi = mutual_info / entropy_x
    proba_sum = count.sum(dim=0) / count.sum()
    codebook_perplexity = torch.exp2(-torch.sum(proba_sum * torch.log2(proba_sum + eps)))
    active_codewords = int((count.sum(dim=0) > 0).sum())
    return UnitsQuality(
        phone_purity=phone_purity.item(),
        cluster_purity=cluster_purity.item(),
        pnmi=pnmi.item(),
        codebook_perplexity=codebook_perplexity.item(),
        active_codewords=active_codewords,
        bitrate=bitrate(num_units=num_units, sequences=[units for (_, units) in data.values()], frame_step=frame_step),
    )


def proba_phone_code(
    data: dict[str | Path, PhonesAndUnits],
    *,
    num_units: int,
    num_phones: int,
    only_active: bool,
    frame_step: int = 20,
    alignment_step: int = 10,
) -> tuple[torch.Tensor, list[str], torch.Tensor]:
    count, phone_order = build_count_matrix(data, num_units, num_phones, frame_step // alignment_step)
    count_by_unit = count.sum(dim=0, keepdim=True)
    proba = torch.where(count_by_unit != 0, count / count_by_unit, torch.zeros_like(count))
    codes_order, argmax = [], proba.argmax(dim=0)
    for phn_idx in range(len(count)):
        indices = torch.where(argmax == phn_idx)[0]
        codes_order.extend(indices[torch.argsort(proba[phn_idx, indices], descending=True)])
    if only_active:
        codes_order = [code for code in codes_order if count[:, code].sum() > 0]
    return proba[:, codes_order], phone_order, torch.tensor(codes_order)
