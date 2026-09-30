# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
"""The channels-last feature extractor must match the reference Conv1d implementation."""

import dataclasses
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from spidr.config import DinoSRConfig
from spidr.models.components import FeatureExtractor, LayerNorm, get_components

NUM_SAMPLES = 16000
NUM_FRAMES = 49
NUM_FEATURES = 512


def reference_forward(extractor: FeatureExtractor, x: Tensor) -> Tensor:
    """Original implementation: Conv1d in (batch, channel, frame) layout."""
    x = x.unsqueeze(1)
    for layer in extractor.conv_layers:
        x = layer.conv(x)
        if isinstance(layer.layer_norm, LayerNorm):
            norm = layer.layer_norm
            x = F.layer_norm(x.transpose(-2, -1), norm.normalized_shape, norm.weight, norm.bias, norm.eps)
            x = x.transpose(-2, -1)
        elif layer.layer_norm is not None:
            x = layer.layer_norm(x)
        x = F.gelu(x)
    return x.transpose(1, 2)


def make_extractor(mode: str, *, bias: bool, dtype: torch.dtype) -> FeatureExtractor:
    cfg = dataclasses.replace(DinoSRConfig(), extractor_mode=mode, extractor_conv_bias=bias)
    extractor = get_components(cfg)[0]
    for module in extractor.modules():  # Non-trivial affine parameters.
        if isinstance(module, (LayerNorm, nn.GroupNorm)):
            nn.init.normal_(module.weight, 1.0, 0.1)
            nn.init.normal_(module.bias, 0.0, 0.1)
        elif isinstance(module, nn.Conv1d) and module.bias is not None:
            nn.init.normal_(module.bias, 0.0, 0.1)
    return extractor.to(dtype)


def output_and_grads(
    forward: Callable[[FeatureExtractor, Tensor], Tensor],
    extractor: FeatureExtractor,
    x: Tensor,
    grad_output: Tensor,
    *,
    autocast_dtype: torch.dtype | None = None,
) -> dict[str, Tensor]:
    """Output, then the gradients of the input and of every parameter for a fixed upstream gradient."""
    x = x.detach().requires_grad_()
    with torch.autocast(x.device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
        out = forward(extractor, x)
    assert out.shape == grad_output.shape
    expected_dtype = autocast_dtype or x.dtype
    if autocast_dtype is not None and x.device.type == "cuda":
        # CUDA autocast runs normalization in FP32. In layer_norm mode the final
        # block ends in normalization + GELU; group_norm mode ends in a convolution.
        if isinstance(extractor.conv_layers[-1].layer_norm, (LayerNorm, nn.GroupNorm)):
            expected_dtype = torch.float32
    assert out.dtype == expected_dtype
    names, params = zip(*extractor.named_parameters(), strict=True)
    grads = torch.autograd.grad(out, [x, *params], grad_output)
    return dict(zip(["output", "input", *names], [out, *grads], strict=True))


def run_both(mode: str, *, bias: bool, dtype: torch.dtype) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    torch.manual_seed(0)
    extractor = make_extractor(mode, bias=bias, dtype=dtype)
    x = torch.randn(3, NUM_SAMPLES, dtype=dtype)
    grad_output = torch.randn(3, NUM_FRAMES, NUM_FEATURES, dtype=dtype)
    actual = output_and_grads(FeatureExtractor.forward, extractor, x, grad_output)
    expected = output_and_grads(reference_forward, extractor, x, grad_output)
    return actual, expected


@pytest.mark.parametrize("mode", ["layer_norm", "group_norm"])
@pytest.mark.parametrize("bias", [False, True])
def test_feature_extractor_matches_reference_float64(mode: str, bias: bool) -> None:
    """In float64 rounding is negligible, so both implementations must agree almost exactly."""
    actual, expected = run_both(mode, bias=bias, dtype=torch.float64)
    assert actual.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=1e-10, atol=1e-10, msg=name)


@pytest.mark.parametrize("mode", ["layer_norm", "group_norm"])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_feature_extractor_matches_reference_low_precision(mode: str, bias: bool, dtype: torch.dtype) -> None:
    """Check both worst-entry and aggregate error without dividing by near-zero entries."""
    actual, expected = run_both(mode, bias=bias, dtype=dtype)
    assert_numerically_close(actual, expected, mode=mode, dtype=dtype)


def assert_numerically_close(
    actual: dict[str, Tensor], expected: dict[str, Tensor], *, mode: str, dtype: torch.dtype
) -> None:
    # Explicit acceptance budgets, not measured guarantees. RMS also catches widespread errors
    # that a single large reference entry could hide in the maximum-error comparison.
    max_tol, rms_tol = {
        torch.float32: (1e-4, 2e-5),
        torch.float16: (2e-2, 5e-3),
        torch.bfloat16: (1e-1, 4e-2),
    }[dtype]
    assert actual.keys() == expected.keys()
    for name in expected:
        a, e = actual[name].double(), expected[name].double()
        assert a.shape == e.shape, name
        assert actual[name].dtype == expected[name].dtype, name
        assert torch.isfinite(a).all(), name
        assert torch.isfinite(e).all(), name
        if mode == "group_norm" and name == "conv_layers.0.conv.bias":
            # GroupNorm with one channel per group removes any per-channel bias: this gradient is exactly zero in
            # theory, so both sides are pure rounding noise. Check it is negligible next to the weight gradient.
            scale = expected["conv_layers.0.conv.weight"].double().abs().max()
            assert max(a.abs().max(), e.abs().max()) <= max_tol * scale, name
            continue
        delta = a - e
        # An exactly zero reference must agree exactly; no division by zero or arbitrary floor.
        max_error, max_scale = delta.abs().max().item(), e.abs().max().item()
        rms_error, rms_scale = delta.square().mean().sqrt().item(), e.square().mean().sqrt().item()
        assert max_error <= max_tol * max_scale, (
            f"{name}: max error {max_error:.6g}, reference max {max_scale:.6g}, tolerance {max_tol}"
        )
        assert rms_error <= rms_tol * rms_scale, (
            f"{name}: RMS error {rms_error:.6g}, reference RMS {rms_scale:.6g}, tolerance {rms_tol}"
        )


@pytest.fixture
def isolated_compile_cache(compiled: bool) -> Iterator[None]:
    """Each configuration gets its own cache; input sizes within a case share it."""
    if compiled:
        torch.compiler.reset()
    try:
        yield
    finally:
        if compiled:
            torch.compiler.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for pretraining precision coverage")
@pytest.mark.parametrize("mode", ["layer_norm", "group_norm"])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
@pytest.mark.usefixtures("isolated_compile_cache")
def test_feature_extractor_cuda_pretraining(
    mode: str, bias: bool, dtype: torch.dtype, compiled: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FP32 parameters under CUDA autocast, with backward outside autocast as in train.py."""
    if dtype == torch.bfloat16 and torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("Pretraining requires Ampere or newer for BF16")
    # Match the public backend settings in setup_pytorch without its global compiler patches.
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction", True)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", False)
    # Strict FP32 equivalence needs IEEE convolution arithmetic too. Matmul's flag
    # does not disable cuDNN TF32. This deliberately differs from training's default
    # cuDNN policy; mixed-precision cases still exercise FP16/BF16 convolutions.
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    torch.manual_seed(0)
    extractor = make_extractor(mode, bias=bias, dtype=torch.float32).cuda().train()
    actual_forward = FeatureExtractor.forward
    expected_forward = reference_forward
    if compiled:
        # Full graphs prevent silent eager fallback from satisfying the compiled test.
        actual_forward = torch.compile(actual_forward, dynamic=True, fullgraph=True)
        expected_forward = torch.compile(expected_forward, dynamic=True, fullgraph=True)
    autocast_dtype = None if dtype == torch.float32 else dtype
    # Reuse each compiled callable across changing batch and frame dimensions.
    for batch, samples in [(2, 8000), (3, 16000), (2, 12080)]:
        frames = samples
        for layer in extractor.conv_layers:
            frames = (frames - layer.kernel_size) // layer.stride + 1
        x = torch.randn(batch, samples, device="cuda", dtype=torch.float32)
        upstream = torch.randn(batch, frames, NUM_FEATURES, device="cuda", dtype=dtype)
        actual = output_and_grads(actual_forward, extractor, x, upstream, autocast_dtype=autocast_dtype)
        expected = output_and_grads(expected_forward, extractor, x, upstream, autocast_dtype=autocast_dtype)
        assert all(p.dtype == torch.float32 for p in extractor.parameters())
        assert_numerically_close(actual, expected, mode=mode, dtype=dtype)
        if compiled:
            # Also anchor both compiled implementations to the original eager implementation.
            eager = output_and_grads(reference_forward, extractor, x, upstream, autocast_dtype=autocast_dtype)
            assert_numerically_close(actual, eager, mode=mode, dtype=dtype)
            assert_numerically_close(expected, eager, mode=mode, dtype=dtype)


def test_conv1d_checkpoint_resumes_channels_last(tiny_dinosr_config: DinoSRConfig, tmp_path: Path) -> None:
    """Old Conv1d weights and populated AdamW moments resume through the new forward path."""
    torch.manual_seed(0)
    original = get_components(tiny_dinosr_config)[0].double()
    original.channels_last = False
    optimizer = torch.optim.AdamW(original.parameters(), lr=1e-3)
    x = torch.randn(2, 1600, dtype=torch.float64)

    def step(extractor: FeatureExtractor, optim: torch.optim.AdamW) -> None:
        optim.zero_grad(set_to_none=True)
        extractor(x).square().mean().backward()
        optim.step()

    step(original, optimizer)
    path = tmp_path / "conv1d.pt"
    torch.save({"model": original.state_dict(), "optimizer": optimizer.state_dict()}, path)
    resumed = get_components(tiny_dinosr_config)[0].double()
    assert resumed.channels_last
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
    checkpoint = torch.load(path, weights_only=True)
    resumed.load_state_dict(checkpoint["model"], strict=True)
    resumed_optimizer.load_state_dict(checkpoint["optimizer"])
    for old_param, new_param in zip(original.parameters(), resumed.parameters(), strict=True):
        torch.testing.assert_close(old_param, new_param, rtol=0, atol=0)
        for key, value in optimizer.state[old_param].items():
            torch.testing.assert_close(value, resumed_optimizer.state[new_param][key], rtol=0, atol=0)
    step(original, optimizer)
    step(resumed, resumed_optimizer)
    for old_param, new_param in zip(original.parameters(), resumed.parameters(), strict=True):
        torch.testing.assert_close(old_param, new_param, rtol=1e-9, atol=1e-10)
        for key, value in optimizer.state[old_param].items():
            torch.testing.assert_close(value, resumed_optimizer.state[new_param][key], rtol=1e-9, atol=1e-10)
