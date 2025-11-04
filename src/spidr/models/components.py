# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
"""Model components for DinoSR and SpidR."""

import math

import torch
from torch import Tensor, nn
from torch import distributed as dist
from torch.nn import functional as F

from spidr.config import DinoSRConfig


class LayerNorm(nn.LayerNorm):
    """Layer norm with transpose."""

    def forward(self, x: Tensor) -> Tensor:
        x = x.transpose(-2, -1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.transpose(-2, -1)


class ConvLayerBlock(nn.Module):
    """Convolution unit of FeatureExtractor."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        layer_norm: nn.Module | None,
        *,
        bias: bool,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.layer_norm = layer_norm
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, bias=bias)
        self.init_weights()

    def init_weights(self) -> None:
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        return F.gelu(x)


class FeatureExtractor(nn.Module):
    """Extract features from audio."""

    def __init__(self, conv_layers: nn.ModuleList) -> None:
        super().__init__()
        self.conv_layers = conv_layers

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(1)  # (batch, channel==1, frame)
        for layer in self.conv_layers:
            x = layer(x)  # (batch, feature, frame)
        return x.transpose(1, 2)  # (batch, frame, feature)


class FeatureProjection(nn.Module):
    """Projects features to encoder dimension."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_features)
        self.projection = nn.Linear(in_features, out_features)

    def forward(self, x: Tensor) -> Tensor:
        x = self.layer_norm(x)
        return self.projection(x)


class ConvPositionalEmbedding(nn.Module):
    """Positional embedding which is placed at the beginning of Transformer."""

    def __init__(self, embed_dim: int, kernel_size: int, groups: int, depth: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.groups = groups
        self.depth = depth
        self.layer_kernel_size = max(3, kernel_size // depth)
        self.num_remove = 1 if self.layer_kernel_size % 2 == 0 else 0
        self.layer_norm = LayerNorm(embed_dim, elementwise_affine=False)
        padding = self.layer_kernel_size // 2
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(embed_dim, embed_dim, self.layer_kernel_size, padding=padding, groups=groups)
                for _ in range(depth)
            ]
        )

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        mask = (
            torch.ones_like(x, dtype=torch.bool).transpose(-2, -1)
            if attention_mask is None
            else attention_mask[:, :, 0]
        )
        x = x.transpose(-2, -1)
        for conv in self.convs:
            x = x * mask
            x = conv(x)
            if self.num_remove > 0:
                x = x[..., : -self.num_remove]
            x = self.layer_norm(x)
            x = F.gelu(x)
        x = x * mask
        return x.transpose(-2, -1)


class SelfAttention(nn.Module):
    """Multihead Self Attention module."""

    def __init__(self, embed_dim: int, num_heads: int, *, qkv_bias: bool, dropout: float = 0.0) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        batch, seq, dim = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.embed_dim, dim=2)
        k = k.view(batch, seq, self.num_heads, dim // self.num_heads).transpose(1, 2)
        q = q.view(batch, seq, self.num_heads, dim // self.num_heads).transpose(1, 2)
        v = v.view(batch, seq, self.num_heads, dim // self.num_heads).transpose(1, 2)
        dropout = self.dropout if self.training else 0.0
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, dropout_p=dropout)
        x = x.transpose(1, 2).contiguous().view(batch, seq, dim)
        return self.proj(x)


class FeedForward(nn.Module):
    """Layer that follows attention layer in encoder layer."""

    def __init__(self, embed_dim: int, interm_features: int, interm_dropout: float) -> None:
        super().__init__()
        self.intermediate_dense = nn.Linear(embed_dim, interm_features)
        self.intermediate_dropout = nn.Dropout(interm_dropout)
        self.output_dense = nn.Linear(interm_features, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.intermediate_dense(x)
        x = F.gelu(x)
        x = self.intermediate_dropout(x)
        return self.output_dense(x)


class TransformerLayer(nn.Module):
    """Combines multihead self attention and feed forward."""

    def __init__(
        self, attention: SelfAttention, dropout: float, feed_forward: FeedForward, *, layer_norm_first: bool
    ) -> None:
        super().__init__()
        self.attention = attention
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(attention.embed_dim)
        self.layer_norm_first = layer_norm_first
        self.feed_forward = feed_forward
        self.final_layer_norm = nn.LayerNorm(attention.embed_dim)

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        residual = x
        if self.layer_norm_first:
            x = self.layer_norm(x)
        x = self.attention(x, attention_mask=attention_mask)
        x = self.dropout(x)
        x = residual + x
        if self.layer_norm_first:
            residual = x
            layer_result = self.feed_forward(self.final_layer_norm(x))
            x = self.dropout(layer_result)
            x = residual + x
        else:
            x = self.layer_norm(x)
            residual = x
            layer_result = self.feed_forward(x)
            x = self.dropout(layer_result)
            x = self.final_layer_norm(residual + x)
        return x, layer_result


class Transformer(nn.Module):
    """Transformer module with positional convolutional embeddings."""

    def __init__(
        self,
        layers: nn.ModuleList,
        pos_conv_embed: ConvPositionalEmbedding,
        dropout: float,
        layer_drop: float,
        *,
        layer_norm_first: bool,
    ) -> None:
        super().__init__()
        self.pos_conv_embed = pos_conv_embed
        self.layer_norm = nn.LayerNorm(pos_conv_embed.embed_dim)
        self.layer_norm_first = layer_norm_first
        self.layer_drop = layer_drop
        self.dropout = nn.Dropout(dropout)
        self.layers = layers
        self.apply(self.init_weights)

    @staticmethod
    def init_weights(module: nn.Module) -> None:
        if isinstance(module, SelfAttention):
            nn.init.xavier_uniform_(module.qkv.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(module.proj.weight)
            nn.init.constant_(module.proj.bias, 0.0)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        x = x + self.pos_conv_embed(x, attention_mask)
        if self.layer_norm_first:
            x = self.layer_norm(x)
        x = self.dropout(x)
        for layer in self.layers:
            if not (self.training and torch.rand(1) <= self.layer_drop):
                x, _ = layer(x, attention_mask)
        if not self.layer_norm_first:
            x = self.layer_norm(x)
        return x

    def get_intermediate_outputs(
        self, x: Tensor, attention_mask: Tensor | None = None, num_layers: int | None = None
    ) -> list[Tensor]:
        if num_layers is not None and not 0 < num_layers <= len(self.layers):
            exception = f"`num_layers` must be between [1, {len(self.layers)}]"
            raise ValueError(exception)

        ret: list[Tensor] = []
        x = x + self.pos_conv_embed(x, attention_mask)
        if self.layer_norm_first:
            x = self.layer_norm(x)
        x = self.dropout(x)
        for layer in self.layers:
            x_output, layer_result = layer(x, attention_mask)
            if not (self.training and torch.rand(1) <= self.layer_drop):
                x = x_output.clone()
            ret.append(layer_result)
            if num_layers is not None and len(ret) >= num_layers:
                return ret
        return ret


class Codebook(nn.Module):
    def __init__(self, encoder_embed_dim: int, codebook_size: int, codebook_decay: float) -> None:
        super().__init__()
        self.codebook_decay = codebook_decay
        self.codebook_size = codebook_size
        codebook = torch.randn(encoder_embed_dim, codebook_size, dtype=torch.float32).unsqueeze(0)
        codebook = F.instance_norm(codebook).transpose(1, 2).squeeze().contiguous()
        self.codebook: Tensor
        self.counts: Tensor
        self.register_buffer("codebook", codebook)
        self.register_buffer("counts", torch.ones(codebook_size, dtype=torch.float32).contiguous())

    @torch.no_grad()
    def forward(self, target: Tensor) -> Tensor:
        codebook = self.codebook / self.counts.unsqueeze(1)
        labels = torch.cdist(target, codebook, p=2).argmin(1)
        onehot_target = F.one_hot(labels, self.codebook_size).float()
        if self.training:
            self.step(onehot_target, target)
        return onehot_target

    @torch.no_grad()
    def step(self, onehot_target: Tensor, target: Tensor) -> None:
        if self.codebook_decay >= 1.0:
            return
        count = onehot_target.sum(0)
        memory = torch.matmul(onehot_target.t(), target)
        if dist.is_initialized():
            dist.all_reduce(memory)  # Sum of embeddings
            dist.all_reduce(count)  # Total counts
        alpha = torch.ones_like(count).unsqueeze(1)
        alpha[count != 0] = self.codebook_decay
        self.counts = alpha.squeeze(1) * self.counts + (1 - alpha).squeeze(1) * count
        self.codebook = alpha * self.codebook + (1 - alpha) * memory


def get_components(
    cfg: DinoSRConfig,
) -> tuple[FeatureExtractor, FeatureProjection, Transformer, nn.ModuleList, nn.ModuleList]:
    if cfg.extractor_mode not in {"group_norm", "layer_norm"}:
        raise ValueError(cfg.extractor_mode)
    blocks = nn.ModuleList()
    in_channels = 1
    for i, (out_channels, kernel_size, stride) in enumerate(cfg.extractor_conv_layer_config):
        layer_norm: nn.Module | None = None
        if cfg.extractor_mode == "group_norm" and i == 0:
            layer_norm = nn.GroupNorm(num_groups=out_channels, num_channels=out_channels, affine=True)
        elif cfg.extractor_mode == "layer_norm":
            layer_norm = LayerNorm(normalized_shape=out_channels, elementwise_affine=True)
        blocks.append(
            ConvLayerBlock(in_channels, out_channels, kernel_size, stride, layer_norm, bias=cfg.extractor_conv_bias)
        )
        in_channels = out_channels
    feature_extractor = FeatureExtractor(blocks)

    feature_projection = FeatureProjection(cfg.extractor_conv_layer_config[-1][0], cfg.encoder_embed_dim)
    pos_conv = ConvPositionalEmbedding(
        cfg.encoder_embed_dim, cfg.encoder_pos_conv_kernel, cfg.encoder_pos_conv_groups, cfg.encoder_pos_conv_depth
    )
    layers = nn.ModuleList()
    for _ in range(cfg.encoder_num_layers):
        attention = SelfAttention(
            cfg.encoder_embed_dim,
            cfg.encoder_num_heads,
            qkv_bias=cfg.encoder_qkv_bias,
            dropout=cfg.encoder_attention_dropout,
        )
        feed_forward = FeedForward(
            cfg.encoder_embed_dim, cfg.encoder_ff_interm_features, cfg.encoder_ff_interm_dropout
        )
        layers.append(
            TransformerLayer(
                attention, cfg.encoder_dropout, feed_forward, layer_norm_first=cfg.encoder_layer_norm_first
            )
        )
    student = Transformer(
        layers,
        pos_conv,
        cfg.encoder_dropout,
        cfg.encoder_layer_drop,
        layer_norm_first=not cfg.encoder_layer_norm_first,
    )

    heads = nn.ModuleList(
        nn.Sequential(nn.Linear(cfg.encoder_embed_dim, cfg.codebook_size), nn.LogSoftmax(dim=-1))
        for _ in range(cfg.num_codebooks)
    )
    codebooks = nn.ModuleList(
        Codebook(cfg.encoder_embed_dim, cfg.codebook_size, cfg.codebook_decay) for _ in range(cfg.num_codebooks)
    )
    return feature_extractor, feature_projection, student, heads, codebooks
