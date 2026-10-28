# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
"""Model loading utilities."""

import json
import warnings
from collections import OrderedDict
from itertools import product
from pathlib import Path
from typing import Any

import torch
from torch.hub import load_state_dict_from_url
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from spidr.config import DinoSRConfig, ModelType, SpidRConfig, model_config_class
from spidr.models.dinosr import DinoSR
from spidr.models.spidr import SpidR


def model_class(cfg: DinoSRConfig) -> type[DinoSR]:
    models: dict[type[DinoSRConfig], type[DinoSR]] = {
        DinoSRConfig: DinoSR,
        SpidRConfig: SpidR,
    }
    if type(cfg) not in models:
        names = [config.__name__ for config in models]
        raise ValueError(f"Unknown model config class {type(cfg).__name__}, expected one of {names}.")
    return models[type(cfg)]


def build_model(
    *,
    cfg: DinoSRConfig | None = None,
    model_type: ModelType | None = None,
    checkpoint: str | Path | None = None,
) -> DinoSR:
    """Build a model from `cfg`, or from the config.json next to `checkpoint`, and load the checkpoint weights."""
    if cfg is None:
        if checkpoint is None or model_type is None:
            raise ValueError("Either `cfg`, or both `model_type` and `checkpoint`, must be given.")
        config_path = Path(checkpoint).parent / "config.json"
        if config_path.is_file():
            data = json.loads(config_path.read_text(encoding="utf-8"))
            cfg = model_config_class(model_type)(**data.get("model", data))
        else:
            warnings.warn("Config file not found when loading checkpoint. Using default config.", stacklevel=2)
            cfg = model_config_class(model_type)()
    model = model_class(cfg)(cfg)
    if checkpoint is not None:
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state_dict = state_dict.get("model", state_dict)
        consume_prefix_in_state_dict_if_present(state_dict, "module.")
        model.load_state_dict(state_dict)
    return model


def state_dict_from_dinosr_fairseq_checkpoint(checkpoint: dict[str, Any]) -> OrderedDict[str, torch.Tensor]:
    mapping = {
        "mask_emb": "mask_embedding",
        "layer_norm.bias": "feature_projection.layer_norm.bias",
        "layer_norm.weight": "feature_projection.layer_norm.weight",
        "post_extract_proj.bias": "feature_projection.projection.bias",
        "post_extract_proj.weight": "feature_projection.projection.weight",
        "encoder.layer_norm.bias": "student.layer_norm.bias",
        "encoder.layer_norm.weight": "student.layer_norm.weight",
    }
    for i in range(7):
        mapping |= {
            f"feature_extractor.conv_layers.{i}.0.weight": f"feature_extractor.conv_layers.{i}.conv.weight",
            f"feature_extractor.conv_layers.{i}.2.1.bias": f"feature_extractor.conv_layers.{i}.layer_norm.bias",
            f"feature_extractor.conv_layers.{i}.2.1.weight": f"feature_extractor.conv_layers.{i}.layer_norm.weight",
        }
    for i in range(5):
        mapping |= {
            f"encoder.pos_conv.{i}.0.bias": f"student.pos_conv_embed.convs.{i}.bias",
            f"encoder.pos_conv.{i}.0.weight": f"student.pos_conv_embed.convs.{i}.weight",
        }
    for i in range(8):
        mapping |= {
            f"_codebook{i}": f"codebooks.{i}.codebook",
            f"_codebook_cnts{i}": f"codebooks.{i}.counts",
            f"heads.{i}.bias": f"heads.{i}.0.bias",
            f"heads.{i}.weight": f"heads.{i}.0.weight",
        }
    for i in range(12):
        fairseq, mine = "encoder.layers", "student.layers"
        mapping |= {
            f"{fairseq}.{i}.fc1.bias": f"{mine}.{i}.feed_forward.intermediate_dense.bias",
            f"{fairseq}.{i}.fc1.weight": f"{mine}.{i}.feed_forward.intermediate_dense.weight",
            f"{fairseq}.{i}.fc2.bias": f"{mine}.{i}.feed_forward.output_dense.bias",
            f"{fairseq}.{i}.fc2.weight": f"{mine}.{i}.feed_forward.output_dense.weight",
            f"{fairseq}.{i}.final_layer_norm.bias": f"{mine}.{i}.final_layer_norm.bias",
            f"{fairseq}.{i}.final_layer_norm.weight": f"{mine}.{i}.final_layer_norm.weight",
            f"{fairseq}.{i}.self_attn.k_proj.bias": f"{mine}.{i}.attention.k_proj.bias",
            f"{fairseq}.{i}.self_attn.k_proj.weight": f"{mine}.{i}.attention.k_proj.weight",
            f"{fairseq}.{i}.self_attn.out_proj.bias": f"{mine}.{i}.attention.proj.bias",
            f"{fairseq}.{i}.self_attn.out_proj.weight": f"{mine}.{i}.attention.proj.weight",
            f"{fairseq}.{i}.self_attn.q_proj.bias": f"{mine}.{i}.attention.q_proj.bias",
            f"{fairseq}.{i}.self_attn.q_proj.weight": f"{mine}.{i}.attention.q_proj.weight",
            f"{fairseq}.{i}.self_attn.v_proj.bias": f"{mine}.{i}.attention.v_proj.bias",
            f"{fairseq}.{i}.self_attn.v_proj.weight": f"{mine}.{i}.attention.v_proj.weight",
            f"{fairseq}.{i}.self_attn_layer_norm.bias": f"{mine}.{i}.layer_norm.bias",
            f"{fairseq}.{i}.self_attn_layer_norm.weight": f"{mine}.{i}.layer_norm.weight",
        }

    ema_mapping = {
        "layer_norm.bias": "teacher.layer_norm.bias",
        "layer_norm.weight": "teacher.layer_norm.weight",
    }
    for i in range(5):
        ema_mapping |= {
            f"pos_conv.{i}.0.bias": f"teacher.pos_conv_embed.convs.{i}.bias",
            f"pos_conv.{i}.0.weight": f"teacher.pos_conv_embed.convs.{i}.weight",
        }
    for i in range(12):
        fairseq, mine = "layers", "teacher.layers"
        ema_mapping |= {
            f"{fairseq}.{i}.fc1.bias": f"{mine}.{i}.feed_forward.intermediate_dense.bias",
            f"{fairseq}.{i}.fc1.weight": f"{mine}.{i}.feed_forward.intermediate_dense.weight",
            f"{fairseq}.{i}.fc2.bias": f"{mine}.{i}.feed_forward.output_dense.bias",
            f"{fairseq}.{i}.fc2.weight": f"{mine}.{i}.feed_forward.output_dense.weight",
            f"{fairseq}.{i}.final_layer_norm.bias": f"{mine}.{i}.final_layer_norm.bias",
            f"{fairseq}.{i}.final_layer_norm.weight": f"{mine}.{i}.final_layer_norm.weight",
            f"{fairseq}.{i}.self_attn.k_proj.bias": f"{mine}.{i}.attention.k_proj.bias",
            f"{fairseq}.{i}.self_attn.k_proj.weight": f"{mine}.{i}.attention.k_proj.weight",
            f"{fairseq}.{i}.self_attn.out_proj.bias": f"{mine}.{i}.attention.proj.bias",
            f"{fairseq}.{i}.self_attn.out_proj.weight": f"{mine}.{i}.attention.proj.weight",
            f"{fairseq}.{i}.self_attn.q_proj.bias": f"{mine}.{i}.attention.q_proj.bias",
            f"{fairseq}.{i}.self_attn.q_proj.weight": f"{mine}.{i}.attention.q_proj.weight",
            f"{fairseq}.{i}.self_attn.v_proj.bias": f"{mine}.{i}.attention.v_proj.bias",
            f"{fairseq}.{i}.self_attn.v_proj.weight": f"{mine}.{i}.attention.v_proj.weight",
            f"{fairseq}.{i}.self_attn_layer_norm.bias": f"{mine}.{i}.layer_norm.bias",
            f"{fairseq}.{i}.self_attn_layer_norm.weight": f"{mine}.{i}.layer_norm.weight",
        }
    new_state_dict = OrderedDict({mapping[k]: v for k, v in checkpoint["model"].items() if k != "_ema"})
    new_state_dict |= {ema_mapping[k]: v for k, v in checkpoint["model"]["_ema"].items()}
    keys_to_remove = []
    for m, i in product(["student.layers", "teacher.layers"], range(12)):
        new_state_dict[f"{m}.{i}.attention.qkv.weight"] = torch.cat(
            [
                new_state_dict[f"{m}.{i}.attention.q_proj.weight"],
                new_state_dict[f"{m}.{i}.attention.k_proj.weight"],
                new_state_dict[f"{m}.{i}.attention.v_proj.weight"],
            ],
            dim=0,
        )
        new_state_dict[f"{m}.{i}.attention.qkv.bias"] = torch.cat(
            [
                new_state_dict[f"{m}.{i}.attention.q_proj.bias"],
                new_state_dict[f"{m}.{i}.attention.k_proj.bias"],
                new_state_dict[f"{m}.{i}.attention.v_proj.bias"],
            ]
        )
        keys_to_remove += [
            f"{m}.{i}.attention.q_proj.weight",
            f"{m}.{i}.attention.k_proj.weight",
            f"{m}.{i}.attention.v_proj.weight",
            f"{m}.{i}.attention.q_proj.bias",
            f"{m}.{i}.attention.k_proj.bias",
            f"{m}.{i}.attention.v_proj.bias",
        ]
    for k in keys_to_remove:
        del new_state_dict[k]
    try:
        current_step = checkpoint["last_optimizer_state"]["state"][0]["step"]
    except KeyError:
        current_step = 0
    new_state_dict["current_step"] = torch.tensor([current_step])
    return new_state_dict


def spidr_base(*, pretrained: bool = True, check_hash: bool = False, progress: bool = True) -> SpidR:
    model = SpidR(SpidRConfig())
    if pretrained:
        url = "https://dl.fbaipublicfiles.com/shared/devai/assets/models/spidr_base.pt"
        checkpoint = load_state_dict_from_url(url, check_hash=check_hash, progress=progress, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def dinosr_base_reproduced(*, pretrained: bool = True, check_hash: bool = False, progress: bool = True) -> DinoSR:
    model = DinoSR(DinoSRConfig())
    if pretrained:
        url = "https://dl.fbaipublicfiles.com/shared/devai/assets/models/dinosr_base_reproduced.pt"
        checkpoint = load_state_dict_from_url(url, check_hash=check_hash, progress=progress, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def dinosr_base_original(*, pretrained: bool = True, check_hash: bool = False, progress: bool = True) -> DinoSR:
    model = DinoSR(DinoSRConfig(encoder_qkv_bias=True))
    if pretrained:
        url = "https://data.csail.mit.edu/placesaudio/dinosr/dinosr.ckpt"
        state_dict = load_state_dict_from_url(url, check_hash=check_hash, progress=progress)
        state_dict = state_dict_from_dinosr_fairseq_checkpoint(state_dict)
        model.load_state_dict(state_dict)
    model.eval()
    return model
