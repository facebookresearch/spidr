# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
"""Tests for DinoSR and SpidR: forward, backward, EMA, codebook updates and layer drop."""

import dataclasses
from collections.abc import Callable
from typing import cast

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from torch import nn

from spidr.config import DinoSRConfig, SpidRConfig
from spidr.models import DinoSR, SpidR
from spidr.models.components import (
    Codebook,
    Codebooks,
    ConvPositionalEmbedding,
    get_components,
    mask_from_index,
    select_masked,
)
from spidr.tools import AverageMeters

BATCH_SIZE = 2
NUM_SAMPLES = 1600
NUM_FRAMES = 78  # (1600 - 10) // 5 + 1 = 319 frames, then (319 - 8) // 4 + 1 = 78.
NUM_MASKED = 15


def make_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    waveforms = torch.randn(BATCH_SIZE, NUM_SAMPLES)
    mask_indices = torch.arange(5, 5 + NUM_MASKED).expand(BATCH_SIZE, -1).contiguous()
    return waveforms, mask_indices


@pytest.mark.parametrize("model_cls", [DinoSR, SpidR])
def test_forward_backward(model_cls: type[DinoSR], request: pytest.FixtureRequest) -> None:
    cfg = request.getfixturevalue("tiny_dinosr_config" if model_cls is DinoSR else "tiny_spidr_config")
    model = model_cls(cfg).train()
    waveforms, mask_indices = make_inputs()
    losses, outputs = model(waveforms, mask_indices=mask_indices, attention_mask=None)
    assert losses.shape == (BATCH_SIZE * NUM_MASKED,)
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
    waveforms, mask_indices = make_inputs()
    model(waveforms, mask_indices=mask_indices)[0].mean().backward()
    without_grad = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert without_grad == []


@pytest.mark.parametrize("model_cls", [DinoSR, SpidR])
def test_perplexities_are_scalars_the_meters_accept(model_cls: type[DinoSR], request: pytest.FixtureRequest) -> None:
    """AverageMeter accumulates into a 0-dim tensor, so the metrics must be 0-dim too."""
    cfg = request.getfixturevalue("tiny_dinosr_config" if model_cls is DinoSR else "tiny_spidr_config")
    waveforms, mask_indices = make_inputs()
    _, outputs = model_cls(cfg).train()(waveforms, mask_indices=mask_indices)
    assert outputs["target_ppl"].shape == ()
    assert outputs["pred_ppl"].shape == ()
    meters = AverageMeters(["target_ppl", "pred_ppl"], device=torch.device("cpu"))
    meters.update(target_ppl=outputs["target_ppl"], pred_ppl=outputs["pred_ppl"])
    assert set(meters.pop()) == {"target_ppl", "pred_ppl"}


@given(
    batch=st.integers(min_value=1, max_value=6),
    frames=st.integers(min_value=1, max_value=40),
    dim=st.integers(min_value=1, max_value=8),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    data=st.data(),
)
@settings(deadline=None)
def test_select_masked_matches_indexing_with_nonzero(
    batch: int, frames: int, dim: int, seed: int, data: st.DataObject
) -> None:
    """`select_masked` replaced `x[torch.nonzero(mask, as_tuple=True)]`; the two must not differ."""
    num_masked = data.draw(st.integers(min_value=0, max_value=frames))
    generator = torch.Generator().manual_seed(seed)
    index = torch.stack(
        [torch.randperm(frames, generator=generator)[:num_masked].sort().values for _ in range(batch)]
    ).reshape(batch, num_masked)
    x = torch.randn(batch, frames, dim, generator=generator)

    mask = mask_from_index(index, frames)
    assert torch.equal(mask.sum(1), torch.full((batch,), num_masked))
    assert torch.equal(torch.nonzero(mask, as_tuple=True)[1].reshape(batch, num_masked), index)
    assert torch.equal(select_masked(x, index), x[torch.nonzero(mask, as_tuple=True)])


def capture_dynamo_graphs(fn: Callable[..., object], *args: object, **kwargs: object) -> tuple[list[str], object]:
    """Run `fn` under `torch.compile` and return the operators Dynamo traced, plus the result.

    `fullgraph=True` alone would not prove there is no host sync: Dynamo traces `torch.nonzero` into
    a single graph behind an unbacked symbol, and the sync then happens at runtime.
    """
    targets: list[str] = []

    def backend(gm: torch.fx.GraphModule, _: object) -> object:
        targets.extend(str(node.target) for node in gm.graph.nodes if node.op in ("call_function", "call_method"))
        return gm.forward

    result = torch.compile(fn, backend=backend, fullgraph=True)(*args, **kwargs)
    return targets, result


SYNCING_OPS = ("nonzero", "_local_scalar_dense", "method 'item'", "function item")


class RecordTorchOps(torch.overrides.TorchFunctionMode):
    """Record the name of every torch operator dispatched inside the block."""

    def __init__(self) -> None:
        self.names: list[str] = []

    def __torch_function__(self, func: object, types: object, args: tuple = (), kwargs: dict | None = None) -> object:
        self.names.append(getattr(func, "__name__", str(func)))
        return func(*args, **(kwargs or {}))  # ty: ignore[call-non-callable]


def test_forward_does_not_sync_with_the_host(tiny_spidr_config: SpidRConfig) -> None:
    """The masked frames come from precomputed positions, so `torch.nonzero` no longer syncs here.

    The output is compared against eager too, to catch a rewrite that traces cleanly but selects the
    wrong frames.
    """
    waveforms, mask_indices = make_inputs()
    model = SpidR(tiny_spidr_config).eval()
    with torch.no_grad():
        expected = model(waveforms, mask_indices=mask_indices)[0]
        targets, actual = capture_dynamo_graphs(model, waveforms, mask_indices=mask_indices)

    assert [target for target in targets if any(op in target for op in SYNCING_OPS)] == []
    # Boolean indexing would show up as a bare `getitem` and calls `nonzero` underneath, so require
    # one of the gathers whose output shape is known up front.
    assert any(op in target for target in targets for op in ("index_select", "take_along_dim", "gather"))
    torch.testing.assert_close(cast("tuple[torch.Tensor, ...]", actual)[0], expected)


@pytest.mark.parametrize("model_cls", [DinoSR, SpidR])
def test_forward_without_a_mask_predicts_every_frame(model_cls: type[DinoSR], request: pytest.FixtureRequest) -> None:
    """`mask_indices=None` selects every frame without substituting the mask embedding."""
    cfg = request.getfixturevalue("tiny_dinosr_config" if model_cls is DinoSR else "tiny_spidr_config")
    waveforms, _ = make_inputs()
    model = model_cls(cfg).eval()
    with torch.no_grad():
        unmasked = model(waveforms)[0]
        every_frame_masked = model(waveforms, mask_indices=torch.arange(NUM_FRAMES).expand(BATCH_SIZE, -1))[0]
    assert unmasked.shape == (BATCH_SIZE * NUM_FRAMES,)
    assert torch.isfinite(unmasked).all()
    assert not torch.allclose(unmasked, every_frame_masked)  # The embedding is substituted, not skipped.


def reference_update_ema(model: DinoSR, step: int, decay: float) -> None:
    """The per-parameter EMA loop that `DinoSR.update_ema` replaced, kept verbatim as an oracle."""
    model.current_step.fill_(step)
    if not 0.0 < decay < 1.0:
        return
    with torch.no_grad():
        for (name, ema_p), model_p in zip(model.teacher.named_parameters(), model.student.parameters(), strict=True):
            if name in model.teacher_exclude_layers:
                ema_p.copy_(model_p)
            else:
                ema_p.lerp_(model_p, 1 - decay)
        for ema_b, model_b in zip(model.teacher.buffers(), model.student.buffers(), strict=True):
            ema_b.copy_(model_b)


def jitter_student(model: DinoSR, generator: torch.Generator) -> None:
    with torch.no_grad():
        for param in model.student.parameters():
            param.add_(torch.randn(param.shape, generator=generator) * 0.01)


@pytest.mark.parametrize("model_cls", [DinoSR, SpidR])
def test_update_ema_matches_the_per_parameter_reference(
    model_cls: type[DinoSR], request: pytest.FixtureRequest
) -> None:
    """The `_foreach` update must be bit-identical to the loop it replaced, over a whole schedule.

    The schedules are shortened so the run covers the ramp, the hold, and the frozen tail.
    """
    cfg = request.getfixturevalue("tiny_dinosr_config" if model_cls is DinoSR else "tiny_spidr_config")
    cfg = (
        dataclasses.replace(cfg, ema_final_step=6, freeze_step=11)
        if model_cls is DinoSR
        else dataclasses.replace(cfg, ema_timescale=3.0, ema_threshold=1e-4)
    )
    torch.manual_seed(0)
    actual = model_cls(cfg)
    torch.manual_seed(0)
    expected = model_cls(cfg)
    generator = torch.Generator().manual_seed(0)

    for step in range(1, 16):
        jitter_student(actual, generator)
        with torch.no_grad():  # Keep both students identical; only the teacher update is under test.
            for target, source in zip(expected.student.parameters(), actual.student.parameters(), strict=True):
                target.copy_(source)
        decay = actual.update_ema(step)
        reference_update_ema(expected, step, decay)
        for i, (a, b) in enumerate(zip(actual.teacher.parameters(), expected.teacher.parameters(), strict=True)):
            assert torch.equal(a, b), f"teacher parameter {i} diverged at step {step}"
        assert torch.equal(actual.current_step, expected.current_step)


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


def test_update_ema_still_targets_the_teacher_after_load_state_dict(tiny_dinosr_config: DinoSRConfig) -> None:
    """`update_ema` writes through tensor lists cached at construction, which resuming must not stale.

    `load_state_dict` copies into the existing parameters rather than rebinding them, so the cache
    stays valid -- but writing into orphaned tensors would silently freeze the teacher on resume.
    """
    model = DinoSR(tiny_dinosr_config).train()
    other = DinoSR(dataclasses.replace(tiny_dinosr_config, encoder_dropout=0.0)).train()
    model.load_state_dict(other.state_dict())

    teacher_before = [p.clone() for p in model.teacher.parameters()]
    jitter_student(model, torch.Generator().manual_seed(0))
    model.update_ema(step=1)
    assert any(
        not torch.equal(before, after)
        for before, after in zip(teacher_before, model.teacher.parameters(), strict=True)
    )


def test_ema_targets_stay_out_of_the_state_dict(tiny_dinosr_config: DinoSRConfig) -> None:
    """`update_ema` caches plain tensor lists, which must not change the checkpoint format."""
    model = DinoSR(tiny_dinosr_config)
    keys = set(model.state_dict())
    model.update_ema(step=1)
    assert set(model.state_dict()) == keys


def test_update_ema_issues_the_same_operations_whatever_the_model_size(tiny_spidr_config: SpidRConfig) -> None:
    """Two `_foreach` calls cover the whole teacher, so the op sequence must not grow with it.

    The loop this replaced issued one op per parameter -- a few hundred kernels per step on a
    base-size model. The schedule runs on the host, so it must also read nothing back from the device.
    """
    shallow = SpidR(tiny_spidr_config).train()
    deep = SpidR(dataclasses.replace(tiny_spidr_config, encoder_num_layers=6)).train()
    assert len(list(deep.teacher.parameters())) > 2 * len(list(shallow.teacher.parameters()))

    with RecordTorchOps() as shallow_ops:
        shallow.update_ema(step=1)
    with RecordTorchOps() as deep_ops:
        deep.update_ema(step=1)

    assert shallow_ops.names == deep_ops.names
    assert shallow_ops.names.count("_foreach_lerp_") == 1  # The averaged half.
    assert shallow_ops.names.count("_foreach_copy_") == 1  # The excluded layers and the buffers.
    assert "lerp_" not in shallow_ops.names
    assert "copy_" not in shallow_ops.names
    assert not {"nonzero", "item", "tolist", "_local_scalar_dense"} & set(shallow_ops.names)


def test_ema_update_freezes_extractor(tiny_dinosr_config: DinoSRConfig) -> None:
    model = DinoSR(tiny_dinosr_config).train()
    assert all(p.requires_grad for p in model.feature_extractor.parameters())
    decay = model.update_ema(step=model.freeze_step)
    assert decay == 1
    assert model._extractor_frozen
    assert all(not p.requires_grad for p in model.feature_extractor.parameters())
    assert all(not p.requires_grad for p in model.feature_projection.parameters())


def test_codebooks_update_in_train_but_not_eval(tiny_spidr_config: SpidRConfig) -> None:
    waveforms, mask_indices = make_inputs()
    model = SpidR(tiny_spidr_config).train()
    before = [codebook.codebook.clone() for codebook in model.codebooks]
    model(waveforms, mask_indices=mask_indices)
    assert all(not torch.equal(b, c.codebook) for b, c in zip(before, model.codebooks, strict=True))

    model = model.eval()
    before = [codebook.codebook.clone() for codebook in model.codebooks]
    with torch.no_grad():
        model(waveforms, mask_indices=mask_indices)
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


@given(
    rows=st.integers(min_value=1, max_value=64),  # `cdist` switches to its matmul form above 25 rows
    dim=st.integers(min_value=2, max_value=32),
    size=st.integers(min_value=2, max_value=32),
    scale=st.floats(min_value=0.1, max_value=10.0),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(deadline=None)
def test_codebook_assigns_the_same_codeword_as_cdist(rows: int, dim: int, size: int, scale: float, seed: int) -> None:
    """`Codebook.forward` expands `torch.cdist(target, entries).argmin(1)` into a single matmul.

    Labels are not compared directly: both forms minimise the same quantity, so they can only differ
    on a near-tie, and then either codeword is a nearest one. What must hold is that the codeword
    actually chosen sits at the minimum distance. The tolerance admits a tie resolved either way --
    a genuinely wrong assignment lands far outside it.
    """
    torch.manual_seed(seed)
    codebook = Codebook(dim, size, 0.9)
    target = torch.randn(rows, dim) * scale

    labels = codebook(target).argmax(1)

    entries = (codebook.codebook / codebook.counts.unsqueeze(1)).double()
    distances = torch.cdist(target.double(), entries, p=2)
    chosen = distances.gather(1, labels[:, None]).squeeze(1)
    torch.testing.assert_close(chosen, distances.min(1).values, rtol=1e-5, atol=1e-5)


def test_codebook_assignment_ignores_the_autocast_dtype() -> None:
    """The matmul is wrapped in `autocast(enabled=False)` so assignment stays float32, as `cdist` was.

    Sized so removing that wrapper is caught: at a smaller codebook bfloat16 happens to pick the same
    codewords anyway, but here it moves several of them.
    """
    torch.manual_seed(0)
    codebook = Codebook(256, 256, 0.9)
    target = torch.randn(512, 256)
    outside = codebook(target)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        inside = codebook(target)
    assert torch.equal(outside, inside)


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
