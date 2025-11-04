# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
"""SpidR model implementation."""

import math
from functools import partial

import torch
import torch.nn.functional as F
from torch import Tensor

from spidr.config import SpidRConfig
from spidr.models.dinosr import DinoSR
from spidr.models.metrics import perplexity


def exp_ema_scheduler(step: int, start_decay: float, timescale: float, threshold: float) -> float:
    decay = 1 - (1 - start_decay) * math.exp(-step / timescale)
    return decay if 1 - decay > threshold else 1


class SpidR(DinoSR):
    def __init__(self, cfg: SpidRConfig | None = None) -> None:
        if cfg is None:
            cfg = SpidRConfig()
        super().__init__(cfg)
        self.ema_scheduler = partial(
            exp_ema_scheduler,
            start_decay=cfg.ema_start_decay,
            timescale=cfg.ema_timescale,
            threshold=cfg.ema_threshold,
        )

    def get_codebooks(
        self,
        waveform: Tensor,
        *,
        attention_mask: Tensor | None = None,
        onehot: bool = False,
    ) -> list[Tensor | None]:
        x = self.feature_extractor(waveform)
        x = self.feature_projection(x)
        preds: list[Tensor | None] = [None] * (len(self.student.layers) - self.num_codebooks)
        for i, y in enumerate(self.student.get_intermediate_outputs(x, attention_mask)[-self.num_codebooks :]):
            pred = self.heads[i](y).float().exp().squeeze()
            if onehot:
                pred = F.one_hot(pred.argmax(dim=-1), pred.size(-1))
            preds.append(pred)
        return preds

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
        mask_indices = torch.nonzero(mask, as_tuple=True)
        log_preds = [
            self.heads[i](y[mask_indices])
            for i, y in enumerate(self.student.get_intermediate_outputs(x, attention_mask)[-self.num_codebooks :])
        ]

        with torch.no_grad():
            targets = self.teacher.get_intermediate_outputs(feats, attention_mask)[-self.num_codebooks :]
            targets = [F.instance_norm(tl.float().transpose(1, 2)).transpose(1, 2)[mask_indices] for tl in targets]

        losses = torch.zeros(log_preds[0].shape[0], device=x.device)
        target_ppl, pred_ppl = torch.zeros((), device=x.device), torch.zeros((), device=x.device)
        for i, (log_pred, target) in enumerate(zip(log_preds, targets, strict=True)):
            onehot_target = self.codebooks[i](target)
            target_ppl += perplexity(onehot_target)
            pred_ppl += perplexity(log_pred.exp())
            losses += torch.sum(-onehot_target * log_pred, dim=-1)

        return (losses / self.num_codebooks), {
            "target_ppl": target_ppl / self.num_codebooks,
            "pred_ppl": pred_ppl / self.num_codebooks,
        }
