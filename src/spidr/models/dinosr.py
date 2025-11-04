# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
"""DinoSR model definition."""

import copy
from collections.abc import Iterable
from functools import partial

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from spidr.config import DinoSRConfig
from spidr.models.components import Transformer, get_components
from spidr.models.metrics import perplexity


def ema_scheduler(step: int, start_decay: float, final_decay: float, final_step: int, freeze_step: int) -> float:
    if step < final_step:
        pct = 1 - step / final_step
        return final_decay - (final_decay - start_decay) * pct
    if step < freeze_step:
        return final_decay
    return 1


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

    def train(self, mode: bool = True) -> "DinoSR":  # noqa: FBT001, FBT002
        super().train(mode)
        self.teacher.eval()
        return self

    @property
    def num_codebooks(self) -> int:
        return len(self.codebooks)

    def freeze_extractor(self) -> None:
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        for p in self.feature_projection.parameters():
            p.requires_grad = False
        self._extractor_frozen = True

    @torch.no_grad()
    def _inner_ema(self, decay: torch.Tensor) -> None:
        for (ema_n, ema_p), model_p in zip(self.teacher.named_parameters(), self.student.parameters(), strict=True):
            if ema_n in self.teacher_exclude_layers:
                ema_p.copy_(model_p)
            else:
                ema_p.lerp_(model_p, 1 - decay)
        for ema_b, model_b in zip(self.teacher.buffers(), self.student.buffers(), strict=True):
            ema_b.copy_(model_b)

    def update_ema(self, step: int) -> float:
        self.current_step.fill_(step)
        decay = self.ema_scheduler(step)
        if not self._extractor_frozen and step >= self.freeze_step:
            self.freeze_extractor()
        if 0.0 < decay < 1.0:
            self._inner_ema(torch.tensor(decay, device=self.current_step.device))
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
        self, waveforms: Tensor, *, mask: Tensor | None = None, attention_mask: Tensor | None = None
    ) -> tuple[Tensor, dict[str, Tensor]]:
        feats = self.feature_extractor(waveforms)
        feats = self.feature_projection(feats)
        x = feats.clone()
        x = self.projection_dropout(x)
        if mask is not None:
            x = torch.where(mask.unsqueeze(-1), self.mask_embedding.to(x.dtype).expand_as(x), x)
        else:
            mask = torch.ones((x.shape[0], x.shape[1]), dtype=torch.bool, device=x.device)
        x = self.student(x, attention_mask)
        mask_indices = torch.nonzero(mask, as_tuple=True)
        x = x[mask_indices]

        with torch.no_grad():
            targets = self.teacher.get_intermediate_outputs(feats, attention_mask)[-self.num_codebooks :]
            targets = [F.instance_norm(tl.float().transpose(1, 2)).transpose(1, 2)[mask_indices] for tl in targets]

        losses = torch.zeros(x.shape[0], device=x.device)
        target_ppl, pred_ppl = torch.zeros(1, device=x.device), torch.zeros(1, device=x.device)
        for i, target in enumerate(targets):
            log_pred = self.heads[i](x)
            onehot_target = self.codebooks[i](target)
            target_ppl += perplexity(onehot_target)
            pred_ppl += perplexity(log_pred.exp())
            losses += torch.sum(-onehot_target * log_pred, dim=-1)

        return (losses / self.num_codebooks), {
            "target_ppl": target_ppl / self.num_codebooks,
            "pred_ppl": pred_ppl / self.num_codebooks,
        }
