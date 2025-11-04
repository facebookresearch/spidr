# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
"""Configuration dataclasses."""

import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

SAMPLE_RATE: int = 16_000
DEFAULT_CONV_LAYER_CONFIG: list[tuple[int, int, int]] = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2

MaskingStrategy = Literal["static", "uniform", "normal", "poisson"]
ModelType = Literal["dinosr", "spidr"]


class SlurmConfig(TypedDict):
    account: str
    nodes: int
    gpus_per_node: int
    qos: NotRequired[str]
    time: NotRequired[int]
    cpus_per_task: NotRequired[int]
    mem_per_gpu: NotRequired[str]
    constraint: NotRequired[str]
    partition: NotRequired[str]


@dataclass(frozen=True)
class RunConfig:
    """Training run configuration."""

    workdir: str
    wandb_project: str
    wandb_name: str
    wandb_mode: Literal["online", "offline", "disabled"] = "offline"
    random_seed: int = 0
    use_deterministic: bool = False
    log_interval: int = 200
    save_interval: int = 2_500
    init_ckpt: str | None = None
    compile: bool = True
    keep_latest: int = -1
    model_type: ModelType = "dinosr"
    slurm_validation: SlurmConfig | None = None

    @property
    def dir(self) -> Path:
        return Path(self.workdir).resolve() / self.wandb_project / self.wandb_name


@dataclass(frozen=True)
class DataConfig:
    """Dataloading and data sampling configuration."""

    manifest: str
    enable_padding: bool = False
    rand_crop: bool = True
    normalize: bool = True
    num_buckets: int = 1000
    min_sample_size: int = 2_000
    max_sample_size: int = 320_000
    max_batch_length: int = 3_800_000
    drop_last: bool = False
    num_workers: int = 10
    pin_memory: bool = False
    prefetch_factor: int = 2
    persistent_workers: bool = True
    random_seed: int = 0
    bucket_method: Literal["uniform", "percentile"] = "uniform"
    alignments_path: Path | None = None


@dataclass(frozen=True)
class OptimizerConfig:
    """AdamW and learning rate scheduler configuration."""

    lr: float = 5e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.01
    eps: float = 1e-6
    max_norm: float = 10.0
    dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"

    init_lr_scale: float = 0.01
    final_lr_scale: float = 0.01
    max_steps: int = 400_000

    warmup_steps: int = 12_000
    hold_steps: int = 188_000
    decay_steps: int = 200_000
    rsqrt_timescale: int = 10_000
    rsqrt_shift: int = 0

    scheduler: Literal["tristage", "cosine", "rsqrt", "constant"] = "tristage"

    exclude_from_optimizer: list[str] = field(default_factory=lambda: ["teacher"])
    to_freeze: list[str] = field(default_factory=lambda: ["feature_extractor", "feature_projection"])

    @property
    def mixed_precision(self) -> bool:
        return self.dtype != "float32"


@dataclass(frozen=True)
class DinoSRConfig:
    """DinoSR configuration. Corresponding names in `fairseq` in comments."""

    extractor_mode: Literal["group_norm", "layer_norm"] = "layer_norm"  # model.extractor_mode
    extractor_conv_bias: bool = False  # model.conv_bias
    # model.conv_layer_config
    extractor_conv_layer_config: list[tuple[int, int, int]] = field(default_factory=lambda: DEFAULT_CONV_LAYER_CONFIG)
    encoder_embed_dim: int = 768  # model.encoder_embed_dim
    encoder_projection_dropout: float = 0  # model.dropout_input
    encoder_pos_conv_kernel: int = 95  # model.conv_pos
    encoder_pos_conv_groups: int = 16  # model.conv_pos_groups
    encoder_pos_conv_depth: int = 5  # model.pos_conv_depth
    encoder_num_layers: int = 12  # model.encoder_layers
    encoder_num_heads: int = 12  # model.encoder_attention_heads
    encoder_attention_dropout: float = 0.1  # model.attention_dropout
    encoder_ff_interm_features: int = 3072  # model.encoder_ffn_embed_dim
    encoder_ff_interm_dropout: float = 0.0  # model.activation_dropout
    encoder_dropout: float = 0.1  # model.dropout
    encoder_layer_norm_first: bool = False  # model.layer_norm_first
    encoder_layer_drop: float = 0.05  # model.encoder_layerdrop
    encoder_qkv_bias: bool = False  # WARNING: not in `fairseq`, always True there.

    codebook_size: int = 256  # model.codebook_size
    codebook_decay: float = 0.9  # model.codebook_decay
    num_codebooks: int = 8  # model.average_top_k_layers
    ema_start_decay: float = 0.999  # model.ema_decay
    ema_final_decay: float = 0.9999  # model.ema_end_decay
    ema_final_step: int = 30_000  # model.ema_anneal_end_step
    ema_exclude_layers: list[str] = field(default_factory=lambda: ["pos_conv_embed"])
    freeze_step: int = 200_000  # model.freeze_teacher_step


@dataclass(frozen=True)
class SpidRConfig(DinoSRConfig):
    """SpidR configuration. Corresponding names in `fairseq` in comments."""

    ema_timescale: float = 20_000
    ema_threshold: float = 1e-7
    encoder_layer_drop: float = 0.0  # model.encoder_layerdrop


@dataclass(frozen=True)
class MaskingConfig:
    """Masking configuration. Corresponding names in `fairseq` in comments."""

    mask_prob: float = 0.8  # model.mask_prob
    mask_selection: MaskingStrategy = "static"  # model.mask_selection
    mask_other: float = 0.0  # model.mask_other
    mask_length: int = 10  # model.mask_length
    no_mask_overlap: bool = False  # model.no_mask_overlap
    mask_min_space: int = 1  # model.mask_min_space
    mask_channel_prob: float = 0.0  # model.mask_channel_prob
    mask_channel_selection: MaskingStrategy = "static"  # model.mask_channel_selection
    mask_channel_other: float = 0.0  # model.mask_channel_other
    mask_channel_length: int = 10  # model.mask_channel_length
    no_mask_channel_overlap: bool = False  # model.no_mask_channel_overlap
    mask_channel_min_space: int = 1  # model.mask_channel_min_space


@dataclass(frozen=True)
class Config:
    """Full configuration."""

    run: RunConfig
    data: DataConfig
    model: DinoSRConfig | SpidRConfig
    optimizer: OptimizerConfig
    masking: MaskingConfig
    validation: dict[str, DataConfig]


@dataclass(frozen=True)
class ResumeConfig:
    """Hold information to resume from a checkpoint for validation."""

    step: int
    checkpoint: Path | None
    results: Path


def read_config(path: str | Path) -> Config:
    path = Path(path)
    if path.suffix not in {".json", ".toml"}:
        raise ValueError(f"Unsupported config file format: {path.suffix}. Use .json or .toml.")
    data = (tomllib.loads if path.suffix == ".toml" else json.loads)(Path(path).read_text(encoding="utf-8"))
    run = RunConfig(**data["run"])  # ty: ignore[missing-argument]
    return Config(
        run=run,
        data=DataConfig(**data["data"]),  # ty: ignore[missing-argument]
        model=(DinoSRConfig if run.model_type == "dinosr" else SpidRConfig)(**data.get("model", {})),
        optimizer=OptimizerConfig(**data.get("optimizer", {})),
        masking=MaskingConfig(**data.get("masking", {})),
        validation={k: DataConfig(**v) for k, v in data["validation"].items()} if "validation" in data else {},  # ty: ignore[missing-argument]
    )
