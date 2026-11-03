# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
"""SpidR2 model implementation."""

from functools import partial

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils import parametrize

from spidr.config import EntropySchedule, SpidR2Config
from spidr.models.components import mask_from_index, select_masked
from spidr.models.metrics import perplexities
from spidr.models.spidr import SpidR


def entropy_scheduler(
    step: Tensor,
    weight: float,
    schedule: EntropySchedule,
    constant_steps: int,
    max_steps: int,
    timescale: float,
    final_weight: float,
) -> Tensor:
    if not weight or schedule == "constant":
        return torch.full_like(step, weight, dtype=torch.float32)
    decay = torch.exp(-(step - constant_steps).clamp(min=0).float() / timescale)
    scheduled = (
        (weight * decay).clamp(min=final_weight)  # Constant while `step <= constant_steps`, where `decay == 1`.
        if schedule == "constant_then_exponential_decrease"
        else (final_weight - (final_weight - weight) * decay).clamp(max=final_weight)
    )
    return torch.where(step < max_steps, scheduled, torch.full_like(scheduled, final_weight))


def entropies(probs: list[Tensor], log_probs: list[Tensor], *, eps: float = 1e-8) -> tuple[Tensor, Tensor]:
    """Predictive and codebook entropies of the distributions, summed over the codebooks.

    The predictive entropy `H(P(Z|Y))` averages the entropy of the individual frames, the codebook
    entropy `H(E[P(Z|Y)])` is the entropy of the mean distribution. Both are estimated on the local
    batch only: unlike the perplexities, they are part of the loss and stay differentiable.
    """
    predictive = torch.stack(
        [-(prob * log_prob).sum(-1).mean() for prob, log_prob in zip(probs, log_probs, strict=True)]
    ).sum()
    mean_probs = torch.stack([prob.mean(0) for prob in probs])
    codebook = -(mean_probs * (mean_probs + eps).log()).sum()
    return predictive, codebook


class UnitNorm(nn.Module):
    def forward(self, weight: Tensor) -> Tensor:
        return F.normalize(weight, dim=1)


def get_heads(embed_dim: int, codebook_size: int, num_codebooks: int) -> nn.ModuleList:
    """Heads with a centered LayerNorm and unit-norm rows, initialized uniformly on the sphere."""
    heads = nn.ModuleList()
    for _ in range(num_codebooks):
        linear = nn.Linear(embed_dim, codebook_size)
        nn.init.normal_(linear.weight)
        linear.weight.data = F.normalize(linear.weight.data, dim=1)
        nn.init.zeros_(linear.bias)
        parametrize.register_parametrization(linear, "weight", UnitNorm())
        heads.append(nn.Sequential(nn.LayerNorm(embed_dim, elementwise_affine=False), linear, nn.LogSoftmax(dim=-1)))
    return heads


def drop_codebooks(module: nn.Module, state_dict: dict[str, Tensor], prefix: str, *args: object) -> None:
    for name in [name for name in state_dict if name.startswith(f"{prefix}codebooks.")]:
        del state_dict[name]


class SpidR2(SpidR):
    def __init__(self, cfg: SpidR2Config | None = None) -> None:
        if cfg is None:
            cfg = SpidR2Config()
        super().__init__(cfg)
        self.cfg = cfg
        del self.codebooks
        self.register_load_state_dict_pre_hook(drop_codebooks)
        self.heads = get_heads(cfg.encoder_embed_dim, cfg.codebook_size, cfg.num_codebooks)
        self.predictive_entropy_scheduler = partial(
            entropy_scheduler,
            weight=cfg.predictive_entropy_weight,
            schedule=cfg.predictive_entropy_schedule,
            constant_steps=cfg.predictive_entropy_constant_steps,
            max_steps=cfg.predictive_entropy_max_steps,
            timescale=cfg.predictive_entropy_timescale,
            final_weight=cfg.predictive_entropy_final_weight,
        )
        self.codebook_entropy_scheduler = partial(
            entropy_scheduler,
            weight=cfg.codebook_entropy_weight,
            schedule=cfg.codebook_entropy_schedule,
            constant_steps=cfg.codebook_entropy_constant_steps,
            max_steps=cfg.codebook_entropy_max_steps,
            timescale=cfg.codebook_entropy_timescale,
            final_weight=cfg.codebook_entropy_final_weight,
        )

    @property
    def num_codebooks(self) -> int:
        return len(self.heads)  # `DinoSR` counts `self.codebooks`, which SpidR2 does not have.

    def forward(
        self, waveforms: Tensor, *, mask_indices: Tensor | None = None, attention_mask: Tensor | None = None
    ) -> tuple[Tensor, dict[str, Tensor]]:
        feats = self.feature_extractor(waveforms)
        feats = self.feature_projection(feats)
        x = feats.clone()
        x = self.projection_dropout(x)
        if mask_indices is not None:
            mask = mask_from_index(mask_indices, x.shape[1])
            x = torch.where(mask.unsqueeze(-1), self.mask_embedding.to(x.dtype).expand_as(x), x)
        else:
            mask_indices = torch.arange(x.shape[1], device=x.device).expand(x.shape[0], -1)
        log_preds = [
            self.heads[i](select_masked(y, mask_indices)).float()
            for i, y in enumerate(self.student.get_intermediate_outputs(x, attention_mask)[-self.num_codebooks :])
        ]

        with torch.no_grad():
            targets = self.teacher.get_intermediate_outputs(feats, attention_mask)[-self.num_codebooks :]
            targets = [
                select_masked(F.instance_norm(tl.float().transpose(1, 2)).transpose(1, 2), mask_indices)
                for tl in targets
            ]
        # The heads are shared with the student, and are trained by the entropy regularisation below.
        log_targets = [self.heads[i](target).float() for i, target in enumerate(targets)]

        preds = [log_pred.exp() for log_pred in log_preds]
        soft_targets = [log_target.exp() for log_target in log_targets]
        losses = torch.zeros(log_preds[0].shape[0], device=x.device)
        for log_pred, soft_target in zip(log_preds, soft_targets, strict=True):
            losses += torch.sum(-soft_target.detach() * log_pred, dim=-1)
        losses_no_reg = losses.mean().detach()

        student_predictive, student_codebook = entropies(preds, log_preds)
        teacher_predictive, teacher_codebook = entropies(soft_targets, log_targets)
        step = self.current_step.squeeze()
        predictive_weight = self.predictive_entropy_scheduler(step)
        codebook_weight = self.codebook_entropy_scheduler(step)
        if self.cfg.reg_student:
            losses = losses + predictive_weight * student_predictive - codebook_weight * student_codebook
        if self.cfg.reg_teacher:
            losses = losses + predictive_weight * teacher_predictive - codebook_weight * teacher_codebook

        onehot_targets = [
            F.one_hot(log_target.argmax(dim=-1), log_target.shape[-1]).float() for log_target in log_targets
        ]
        ppls = perplexities(onehot_targets + preds)
        target_ppl, pred_ppl = ppls[: self.num_codebooks].sum(), ppls[self.num_codebooks :].sum()
        return (losses / self.num_codebooks), {
            "target_ppl": target_ppl / self.num_codebooks,
            "pred_ppl": pred_ppl / self.num_codebooks,
            "loss_no_reg": losses_no_reg / self.num_codebooks,
            "student_predictive_entropy": student_predictive.detach() / self.num_codebooks,
            "student_codebook_entropy": student_codebook.detach() / self.num_codebooks,
            "teacher_predictive_entropy": teacher_predictive.detach() / self.num_codebooks,
            "teacher_codebook_entropy": teacher_codebook.detach() / self.num_codebooks,
            "predictive_entropy_weight": predictive_weight,
            "codebook_entropy_weight": codebook_weight,
        }
