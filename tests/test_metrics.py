# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
"""Tests for the coalesced perplexity computation."""

import pytest
import torch

from spidr.models.metrics import NonFiniteError, params_norm, perplexities


def reference_perplexity(y: torch.Tensor) -> torch.Tensor:
    """The single-tensor formula `perplexities` batches over."""
    mean = y.mean(0)
    return torch.exp2((-mean * (mean + 1e-8).log2()).sum())


def make_distributions(count: int, rows: int = 40, size: int = 8) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    ys = [torch.rand(rows, size, generator=generator) for _ in range(count)]
    return [y / y.sum(-1, keepdim=True) for y in ys]


def test_perplexities_matches_the_single_tensor_formula() -> None:
    ys = make_distributions(5)
    torch.testing.assert_close(perplexities(ys), torch.stack([reference_perplexity(y) for y in ys]))


def test_perplexities_of_one_hot_inputs_counts_used_entries() -> None:
    """A codebook using k entries uniformly has perplexity k."""
    labels = torch.arange(40) % 4
    onehot = torch.nn.functional.one_hot(labels, 8).float()
    torch.testing.assert_close(perplexities([onehot]), torch.tensor([4.0]))


def test_perplexities_is_order_preserving() -> None:
    ys = make_distributions(4)
    batched = perplexities(ys)
    assert batched.shape == (4,)
    for i, y in enumerate(ys):
        torch.testing.assert_close(batched[i], perplexities([y])[0])


def test_perplexities_rejects_ragged_inputs() -> None:
    """The row count is read from the first tensor, so a mismatch would divide the rest wrongly."""
    with pytest.raises(ValueError, match="same leading dimension"):
        perplexities([torch.rand(4, 8), torch.rand(5, 8)])


def test_perplexities_does_not_build_a_graph() -> None:
    y = torch.rand(4, 8, requires_grad=True)
    assert not perplexities([y]).requires_grad


def test_params_norm_reports_non_finite() -> None:
    finite = [torch.ones(3), torch.zeros(2)]
    torch.testing.assert_close(params_norm(finite), torch.tensor(3.0).sqrt())
    with pytest.raises(NonFiniteError):
        params_norm([torch.tensor([float("nan")])], error_if_nonfinite=True)
