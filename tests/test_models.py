# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
"""Tests for DinoSR and SpidR: forward, backward, EMA, codebook updates and layer drop."""

import dataclasses
from typing import cast

import pytest
import torch
from torch import nn

from spidr.config import DinoSRConfig, SpidRConfig
from spidr.models import DinoSR, SpidR
from spidr.models.components import Codebook, Codebooks, ConvPositionalEmbedding, get_components
from spidr.tools import AverageMeters

BATCH_SIZE = 2
NUM_SAMPLES = 1600
NUM_FRAMES = 78  # (1600 - 10) // 5 + 1 = 319 frames, then (319 - 8) // 4 + 1 = 78.


def make_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    waveforms = torch.randn(BATCH_SIZE, NUM_SAMPLES)
    mask = torch.zeros(BATCH_SIZE, NUM_FRAMES, dtype=torch.bool)
    mask[:, 5:20] = True
    return waveforms, mask


@pytest.mark.parametrize("model_cls", [DinoSR, SpidR])
def test_forward_backward(model_cls: type[DinoSR], request: pytest.FixtureRequest) -> None:
    cfg = request.getfixturevalue("tiny_dinosr_config" if model_cls is DinoSR else "tiny_spidr_config")
    model = model_cls(cfg).train()
    waveforms, mask = make_inputs()
    losses, outputs = model(waveforms, mask=mask, attention_mask=None)
    assert losses.shape == (int(mask.sum()),)
    assert torch.isfinite(losses).all()
    assert torch.isfinite(outputs["target_ppl"]).all()
    assert torch.isfinite(outputs["pred_ppl"]).all()

    losses.mean().backward()
    assert all(p.grad is not None for p in model.student.parameters() if p.requires_grad)
    assert all(p.grad is None for p in model.teacher.parameters())
    if model_cls is SpidR:
        # The last post-norm layer's final_layer_norm is unreachable by the SpidR loss.
        final_layer_norm = cast("nn.Module", model.student.layers[-1].final_layer_norm)
        assert all(not p.requires_grad for p in final_layer_norm.parameters())
    else:
        assert all(p.requires_grad for p in model.student.parameters())


@pytest.mark.parametrize("model_cls", [DinoSR, SpidR])
def test_every_trainable_parameter_receives_a_gradient(
    model_cls: type[DinoSR], request: pytest.FixtureRequest
) -> None:
    """Without layer drop there must be no unused parameters, or DDP needs find_unused_parameters."""
    cfg = request.getfixturevalue("tiny_dinosr_config" if model_cls is DinoSR else "tiny_spidr_config")
    model = model_cls(dataclasses.replace(cfg, encoder_layer_drop=0.0)).train()
    waveforms, mask = make_inputs()
    model(waveforms, mask=mask)[0].mean().backward()
    without_grad = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert without_grad == []


@pytest.mark.parametrize("model_cls", [DinoSR, SpidR])
def test_perplexities_are_scalars_the_meters_accept(model_cls: type[DinoSR], request: pytest.FixtureRequest) -> None:
    """AverageMeter accumulates into a 0-dim tensor, so the metrics must be 0-dim too."""
    cfg = request.getfixturevalue("tiny_dinosr_config" if model_cls is DinoSR else "tiny_spidr_config")
    waveforms, mask = make_inputs()
    _, outputs = model_cls(cfg).train()(waveforms, mask=mask)
    assert outputs["target_ppl"].shape == ()
    assert outputs["pred_ppl"].shape == ()
    meters = AverageMeters(["target_ppl", "pred_ppl"], device=torch.device("cpu"))
    meters.update(target_ppl=outputs["target_ppl"], pred_ppl=outputs["pred_ppl"])
    assert set(meters.pop()) == {"target_ppl", "pred_ppl"}


def test_ema_update_moves_teacher(tiny_dinosr_config: DinoSRConfig) -> None:
    model = DinoSR(tiny_dinosr_config).train()
    teacher_before = [p.clone() for p in model.teacher.parameters()]
    decay = model.update_ema(step=1)
    assert 0.0 < decay < 1.0
    changed = [
        not torch.equal(before, after)
        for before, after in zip(teacher_before, model.teacher.parameters(), strict=True)
    ]
    assert any(changed)
    assert int(model.current_step) == 1


def test_ema_decay_buffer_stays_out_of_the_state_dict(tiny_dinosr_config: DinoSRConfig) -> None:
    """`_ema_decay` is reused across steps but must not change the checkpoint format."""
    model = DinoSR(tiny_dinosr_config)
    keys = set(model.state_dict())
    model.update_ema(step=1)
    assert set(model.state_dict()) == keys
    assert "_ema_decay" not in keys
    assert "_ema_decay" in dict(model.named_buffers())


def test_ema_update_freezes_extractor(tiny_dinosr_config: DinoSRConfig) -> None:
    model = DinoSR(tiny_dinosr_config).train()
    assert all(p.requires_grad for p in model.feature_extractor.parameters())
    decay = model.update_ema(step=model.freeze_step)
    assert decay == 1
    assert model._extractor_frozen
    assert all(not p.requires_grad for p in model.feature_extractor.parameters())
    assert all(not p.requires_grad for p in model.feature_projection.parameters())


def test_codebooks_update_in_train_but_not_eval(tiny_spidr_config: SpidRConfig) -> None:
    waveforms, mask = make_inputs()
    model = SpidR(tiny_spidr_config).train()
    before = [codebook.codebook.clone() for codebook in model.codebooks]
    model(waveforms, mask=mask)
    assert all(not torch.equal(b, c.codebook) for b, c in zip(before, model.codebooks, strict=True))

    model = model.eval()
    before = [codebook.codebook.clone() for codebook in model.codebooks]
    with torch.no_grad():
        model(waveforms, mask=mask)
    assert all(torch.equal(b, c.codebook) for b, c in zip(before, model.codebooks, strict=True))


def test_codebook_state_dict_keys_are_stable(tiny_spidr_config: SpidRConfig) -> None:
    """`Codebooks` subclasses ModuleList so published checkpoints keep loading."""
    model = SpidR(tiny_spidr_config)
    assert isinstance(model.codebooks, nn.ModuleList)
    keys = [key for key in model.state_dict() if key.startswith("codebooks.")]
    expected = [f"codebooks.{i}.{name}" for i in range(model.num_codebooks) for name in ("codebook", "counts")]
    assert sorted(keys) == sorted(expected)


def test_quantize_matches_updating_each_codebook_separately(tiny_spidr_config: SpidRConfig) -> None:
    """Coalescing the codebook statistics must be a pure rearrangement."""
    torch.manual_seed(0)
    reference = get_components(tiny_spidr_config)[-1].train()
    batched = Codebooks(Codebook(16, 8, 0.9) for _ in range(len(reference)))
    batched.load_state_dict(reference.state_dict())
    batched.train()
    targets = [torch.randn(32, 16) for _ in reference]

    sequential = []
    for codebook, target in zip(reference, targets, strict=True):
        onehot = codebook(target)
        codebook.update(onehot.sum(0), onehot.t() @ target)
        sequential.append(onehot)
    coalesced = batched.quantize(targets)

    assert all(torch.equal(a, b) for a, b in zip(sequential, coalesced, strict=True))
    for expected, actual in zip(reference, batched, strict=True):
        torch.testing.assert_close(expected.codebook, actual.codebook)
        torch.testing.assert_close(expected.counts, actual.counts)


def test_quantize_leaves_frozen_codebooks_untouched() -> None:
    """A codebook with decay >= 1 must not move, even when a sibling triggers the update."""
    torch.manual_seed(0)
    codebooks = Codebooks([Codebook(16, 8, 0.9), Codebook(16, 8, 1.0)]).train()
    before = [codebook.codebook.clone() for codebook in codebooks]
    codebooks.quantize([torch.randn(32, 16), torch.randn(32, 16)])
    moving, frozen = codebooks
    assert not torch.equal(before[0], moving.codebook)
    assert torch.equal(before[1], frozen.codebook)


def test_intermediate_outputs_are_not_mutated_by_later_layers(tiny_dinosr_config: DinoSRConfig) -> None:
    """`get_intermediate_outputs` returns layer tensors without cloning; nothing may write into them."""
    torch.manual_seed(0)
    student = get_components(tiny_dinosr_config)[2].eval()
    x = torch.randn(2, 20, tiny_dinosr_config.encoder_embed_dim)
    for before_residual in (True, False):
        outputs = student.get_intermediate_outputs(x, before_residual=before_residual)
        snapshot = [tensor.clone() for tensor in outputs]
        student.get_intermediate_outputs(x, before_residual=before_residual)
        assert all(torch.equal(a, b) for a, b in zip(snapshot, outputs, strict=True))
        assert len({id(tensor) for tensor in outputs}) == len(outputs)


def test_layer_drop_only_skips_layers_while_training(tiny_dinosr_config: DinoSRConfig) -> None:
    cfg = dataclasses.replace(tiny_dinosr_config, encoder_dropout=0.0, encoder_attention_dropout=0.0)
    x = torch.randn(2, 20, cfg.encoder_embed_dim)
    torch.manual_seed(0)
    never = get_components(dataclasses.replace(cfg, encoder_layer_drop=0.0))[2]
    torch.manual_seed(0)
    always = get_components(dataclasses.replace(cfg, encoder_layer_drop=1.0))[2]

    assert torch.equal(never.train()(x), never.eval()(x))  # Disabled: no layer is ever skipped.
    assert torch.equal(always.eval()(x), never.eval()(x))  # Same weights, and eval never skips.
    assert not torch.equal(always.train()(x), always.eval()(x))  # Training with p=1 skips every layer.


def test_conv_positional_embedding_no_mask_matches_all_ones_mask() -> None:
    torch.manual_seed(0)
    mod = ConvPositionalEmbedding(embed_dim=16, kernel_size=10, groups=4, depth=3).eval()
    x = torch.randn(2, 37, 16)
    all_ones_mask = torch.ones(2, 1, 1, 37, dtype=torch.bool)

    out_without_mask = mod(x, attention_mask=None)
    out_with_all_ones_mask = mod(x, attention_mask=all_ones_mask)

    assert torch.equal(out_without_mask, out_with_all_ones_mask)


def test_teacher_stays_in_eval_mode(tiny_dinosr_config: DinoSRConfig) -> None:
    model = DinoSR(tiny_dinosr_config).train()
    assert model.training
    assert not model.teacher.training
