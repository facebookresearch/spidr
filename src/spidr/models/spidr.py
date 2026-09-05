# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
"""SpidR model implementation."""

from functools import partial

import torch
import torch.nn.functional as F
from torch import Tensor

from spidr.config import SpidRConfig
from spidr.models.components import mask_from_index, select_masked
from spidr.models.dinosr import DinoSR
from spidr.models.metrics import perplexities


def exp_ema_scheduler(step: Tensor, start_decay: float, timescale: float, threshold: float) -> Tensor:
    """Decay approaching 1 exponentially, snapping to it once the teacher would barely move.

    Takes and returns a 0-dim tensor, in float64 then narrowed to float32, for the same reasons as
    `spidr.models.dinosr.ema_scheduler`.
    """
    decay = 1 - (1 - start_decay) * torch.exp(-step.double() / timescale)
    return torch.where(1 - decay > threshold, decay, torch.ones_like(decay)).float()


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
        self.student.get_submodule(  # Last normalization layer of the student, which is never used in the loss.
            "layer_norm" if cfg.encoder_layer_norm_first else f"layers.{cfg.encoder_num_layers - 1}.final_layer_norm"
        ).requires_grad_(requires_grad=False)

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
        log_preds = [
            self.heads[i](select_masked(y, mask_index))
            for i, y in enumerate(self.student.get_intermediate_outputs(x, attention_mask)[-self.num_codebooks :])
        ]

        with torch.no_grad():
            targets = self.teacher.get_intermediate_outputs(feats, attention_mask)[-self.num_codebooks :]
            targets = [
                select_masked(F.instance_norm(tl.float().transpose(1, 2)).transpose(1, 2), mask_index)
                for tl in targets
            ]

        onehot_targets = self.codebooks.quantize(targets)
        losses = torch.zeros(log_preds[0].shape[0], device=x.device)
        for log_pred, onehot_target in zip(log_preds, onehot_targets, strict=True):
            losses += torch.sum(-onehot_target * log_pred, dim=-1)
        ppls = perplexities(onehot_targets + [log_pred.exp() for log_pred in log_preds])
        target_ppl, pred_ppl = ppls[: self.num_codebooks].sum(), ppls[self.num_codebooks :].sum()
        return (losses / self.num_codebooks), {
            "target_ppl": target_ppl / self.num_codebooks,
            "pred_ppl": pred_ppl / self.num_codebooks,
        }
