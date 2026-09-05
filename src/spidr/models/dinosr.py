# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
"""DinoSR model definition."""

import copy
from collections.abc import Iterable
from functools import partial

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from spidr.config import DinoSRConfig
from spidr.models.components import Transformer, get_components, mask_from_index, select_masked
from spidr.models.metrics import perplexities


def ema_scheduler(step: Tensor, start_decay: float, final_decay: float, final_step: int, freeze_step: int) -> Tensor:
    """Ramp the decay from `start_decay` to `final_decay`, hold it, then freeze the teacher.

    Takes and returns a 0-dim tensor: the schedule is evaluated on the model's device, without a
    host round-trip, so that `DinoSR.update_ema` stays compilable. The arithmetic runs in float64
    to match the Python-float schedule this replaced, then narrows to the parameters' float32.
    """
    progress = step.double() / final_step
    ramp = final_decay - (final_decay - start_decay) * (1 - progress)
    held = torch.where(step < freeze_step, torch.full_like(ramp, final_decay), torch.ones_like(ramp))
    return torch.where(step < final_step, ramp, held).float()


def init_teacher(
    student: Transformer, exclude_layers: Iterable[str], *, init_weights: bool = True
) -> tuple[Transformer, set[str]]:
    teacher = copy.deepcopy(student).float()
    if init_weights:
        teacher.apply(teacher.init_weights)
    teacher.eval()
    teacher.requires_grad_(requires_grad=False)
    teacher_exclude_layers: set[str] = set()
    for name, param in teacher.named_parameters():
        param.detach_()
        if any(name.startswith(ex) for ex in exclude_layers):
            teacher_exclude_layers.add(name)
    return teacher, teacher_exclude_layers


def _split_ema_targets(
    teacher: Transformer, student: Transformer, exclude_layers: set[str]
) -> tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor]]:
    """Pair every teacher tensor with its student counterpart, split by how the EMA treats it.

    The first pair of lists is averaged; the second -- excluded layers and buffers -- tracks the
    student exactly. Keeping them apart lets `DinoSR.update_ema` run as two `_foreach` calls with no
    Python branch per tensor. Caching the tensors is safe for the life of the model: `.to()` and
    `load_state_dict` both mutate parameters and buffers in place.
    """
    lerp_teacher, lerp_student, copy_teacher, copy_student = [], [], [], []
    for (name, teacher_param), student_param in zip(teacher.named_parameters(), student.parameters(), strict=True):
        if name in exclude_layers:
            copy_teacher.append(teacher_param)
            copy_student.append(student_param)
        else:
            lerp_teacher.append(teacher_param)
            lerp_student.append(student_param)
    for teacher_buffer, student_buffer in zip(teacher.buffers(), student.buffers(), strict=True):
        if not teacher_buffer.is_floating_point():
            raise TypeError(f"The EMA cannot track the non-floating point teacher buffer {teacher_buffer.dtype}")
        copy_teacher.append(teacher_buffer)
        copy_student.append(student_buffer)
    return lerp_teacher, lerp_student, copy_teacher, copy_student


class DinoSR(nn.Module):
    def __init__(self, cfg: DinoSRConfig | None = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = DinoSRConfig()

        self.feature_extractor, self.feature_projection, self.student, self.heads, self.codebooks = get_components(cfg)
        self.teacher, self.teacher_exclude_layers = init_teacher(self.student, cfg.ema_exclude_layers)
        self.ema_scheduler = partial(
            ema_scheduler,
            start_decay=cfg.ema_start_decay,
            final_decay=cfg.ema_final_decay,
            final_step=cfg.ema_final_step,
            freeze_step=cfg.freeze_step,
        )
        self.projection_dropout = nn.Dropout(cfg.encoder_projection_dropout)
        self.freeze_step = cfg.freeze_step
        self._extractor_frozen = False
        self.mask_embedding = nn.Parameter(torch.FloatTensor(cfg.encoder_embed_dim))
        nn.init.uniform_(self.mask_embedding)
        self.current_step = nn.Buffer(torch.zeros(1, dtype=torch.int64))
        (
            self._ema_lerp_teacher,
            self._ema_lerp_student,
            self._ema_copy_teacher,
            self._ema_copy_student,
        ) = _split_ema_targets(self.teacher, self.student, self.teacher_exclude_layers)

    def train(self, mode: bool = True) -> "DinoSR":
        super().train(mode)
        self.teacher.eval()
        return self

    @property
    def num_codebooks(self) -> int:
        return len(self.codebooks)

    @property
    def extractor_frozen(self) -> bool:
        return self._extractor_frozen

    def freeze_extractor(self) -> None:
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        for p in self.feature_projection.parameters():
            p.requires_grad = False
        self._extractor_frozen = True

    @torch.no_grad()
    def update_ema(self) -> Tensor:
        """Advance `current_step` by one and move the teacher towards the student. Returns the decay.

        Every operation runs on the model's device over tensors of known shape -- no host sync, no
        Python branch, no per-parameter kernel -- so the whole update can be `torch.compile`d. The
        caller owns the step counter only in the sense that it must call this exactly once per
        optimizer step; the value itself lives in the `current_step` buffer and survives checkpoints.

        Outside `0 < decay < 1` the teacher is left untouched, matching the schedules' endpoints: a
        decay of 1 freezes it and a decay of 0 is treated as "not yet running". Both fall out of the
        lerp weights rather than a branch, and `lerp` is exact at weight 0 and 1.
        """
        self.current_step.add_(1)  # In place: `+=` would re-register the buffer and break the graph.
        decay = self.ema_scheduler(self.current_step.reshape(()))
        running = (decay > 0) & (decay < 1)
        if self._ema_lerp_teacher:
            weight = torch.where(running, 1 - decay, torch.zeros_like(decay))
            # The stubs only know the scalar weights; the tensor overload keeps the weight on device.
            torch._foreach_lerp_(self._ema_lerp_teacher, self._ema_lerp_student, weight)  # ty: ignore[no-matching-overload]
        if self._ema_copy_teacher:  # A weight of exactly 1 makes `lerp` a copy, and 0 a no-op.
            torch._foreach_lerp_(self._ema_copy_teacher, self._ema_copy_student, running.to(decay.dtype))  # ty: ignore[no-matching-overload]
        return decay

    def get_intermediate_outputs(self, waveforms: Tensor, *, attention_mask: Tensor | None = None) -> list[Tensor]:
        x = self.feature_extractor(waveforms)
        x = self.feature_projection(x)
        return self.student.get_intermediate_outputs(x, attention_mask)

    def get_codebooks(
        self,
        waveform: Tensor,
        *,
        attention_mask: Tensor | None = None,
        onehot: bool = False,
    ) -> list[Tensor | None]:
        x = self.feature_extractor(waveform)
        x = self.feature_projection(x)
        x = self.student(x, attention_mask)
        codebooks: list[Tensor | None] = [None] * (len(self.student.layers) - self.num_codebooks)
        for i in range(self.num_codebooks):
            codebook = self.heads[i](x).float().exp().squeeze()
            if onehot:
                codebook = F.one_hot(codebook.argmax(dim=-1), codebook.size(-1))
            codebooks.append(codebook)
        return codebooks

    def forward(
        self, waveforms: Tensor, *, mask_index: Tensor | None = None, attention_mask: Tensor | None = None
    ) -> tuple[Tensor, dict[str, Tensor]]:
        feats = self.feature_extractor(waveforms)
        feats = self.feature_projection(feats)
        x = feats.clone()
        x = self.projection_dropout(x)
        if mask_index is not None:
            mask = mask_from_index(mask_index, x.shape[1])
            x = torch.where(mask.unsqueeze(-1), self.mask_embedding.to(x.dtype).expand_as(x), x)
        else:  # Nothing is masked out, so every frame is predicted.
            mask_index = torch.arange(x.shape[1], device=x.device).expand(x.shape[0], -1)
        x = self.student(x, attention_mask)
        x = select_masked(x, mask_index)

        with torch.no_grad():
            targets = self.teacher.get_intermediate_outputs(feats, attention_mask)[-self.num_codebooks :]
            targets = [
                select_masked(F.instance_norm(tl.float().transpose(1, 2)).transpose(1, 2), mask_index)
                for tl in targets
            ]

        onehot_targets = self.codebooks.quantize(targets)
        log_preds = [self.heads[i](x) for i in range(self.num_codebooks)]
        losses = torch.zeros(x.shape[0], device=x.device)
        for onehot_target, log_pred in zip(onehot_targets, log_preds, strict=True):
            losses += torch.sum(-onehot_target * log_pred, dim=-1)
        ppls = perplexities(onehot_targets + [log_pred.exp() for log_pred in log_preds])
        target_ppl, pred_ppl = ppls[: self.num_codebooks].sum(), ppls[self.num_codebooks :].sum()
        return (losses / self.num_codebooks), {
            "target_ppl": target_ppl / self.num_codebooks,
            "pred_ppl": pred_ppl / self.num_codebooks,
        }
