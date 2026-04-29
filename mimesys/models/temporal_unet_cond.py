from functools import partial
from typing import Union
from collections import defaultdict, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from einops import rearrange
from einops.layers.torch import Rearrange
from omegaconf import DictConfig, OmegaConf

from mimesys.models.utils import (
    DownSample,
    PreNorm,
    Residual,
    RMSNorm,
    SelfAttentionBlock,
    SinusoidalPosEmbedding,
    LearnableFourierEmbedding,
    UpSample,
)


class ContextEncoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(ContextEncoder, self).__init__()
        self.cfg = cfg

        self.encoder = nn.Sequential(
            nn.Linear(self.cfg.input_dim, self.cfg.hidden_dim),
            nn.Mish(),
            nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim),
            nn.Mish(),
            nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim),
            nn.Mish(),
            nn.Linear(self.cfg.hidden_dim, self.cfg.context_dim)
        )

    def forward(self, x):
        return self.encoder(x)

class TransformerContextEncoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(TransformerContextEncoder, self).__init__()
        self.cfg = cfg
        self.metric_encoder = nn.Linear(self.cfg.input_dim, self.cfg.hidden_dim)
        self.action_encoder = nn.Linear(self.cfg.action_dim, self.cfg.hidden_dim)

        self.encoder = nn.Sequential(
            nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.cfg.hidden_dim,
                nhead=self.cfg.num_heads,
                dim_feedforward=self.cfg.hidden_dim * 4,
                dropout=self.cfg.dropout,
                activation="gelu",
            ),
            num_layers=self.cfg.num_layers,
            ),
            nn.Linear(self.cfg.hidden_dim, self.cfg.context_dim),
        )

    def forward(self, context_cond, lam=0.1, sensitivity_drop_p=0.0):
        metric, action = context_cond['metric'], context_cond['prev_action']
        drop_mask = torch.rand_like(metric) > sensitivity_drop_p
        metric = metric * drop_mask
        metric_emb = self.metric_encoder(metric)

        sum_row = action.sum(dim=1)
        sum_col = action.sum(dim=2)
        current_action = torch.cat([sum_row, sum_col], dim=1)

        action_emb = self.action_encoder(current_action)

        # x = torch.cat([metric_emb, action_emb], dim=-1)
        x = (1 - lam) * metric_emb + lam * action_emb
        x = nn.Mish()(x)
        return self.encoder(x)

class MetricsOnlyContextEncoder(nn.Module):
    """Baseline: ignores prev_action, encodes only metrics."""
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.Mish(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.Mish(),
            nn.Linear(cfg.hidden_dim, cfg.context_dim),
        )

    def forward(self, context_cond, **kwargs):
        return self.encoder(context_cond['metric'])


class ConcatContextEncoder(nn.Module):
    """Concat: flatten full action + metrics, project through MLP."""
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(cfg.action_dim + cfg.input_dim, cfg.hidden_dim),
            nn.Mish(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.Mish(),
            nn.Linear(cfg.hidden_dim, cfg.context_dim),
        )

    def forward(self, context_cond, **kwargs):
        metric = context_cond['metric']          # (B, input_dim)
        action = context_cond['prev_action']     # (B, H, C)
        x = torch.cat([action.flatten(1), metric], dim=-1)
        return self.head(x)


class FiLMContextEncoder(nn.Module):
    """FiLM: metrics produce scale+shift to condition full 260-dim action embedding."""
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.action_proj = nn.Linear(cfg.action_dim, cfg.hidden_dim)  # action_dim=260
        self.film_fc     = nn.Linear(cfg.input_dim,  2 * cfg.hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.Mish(),
            nn.Linear(cfg.hidden_dim, cfg.context_dim),
        )

    def forward(self, context_cond, **kwargs):
        metric = context_cond['metric']                 # (B, input_dim)
        action = context_cond['prev_action']            # (B, H, C)
        action_flat = action.flatten(1)                 # (B, H*C = 260)

        ea = F.relu(self.action_proj(action_flat))      # (B, hidden_dim)
        gamma, beta = self.film_fc(metric).chunk(2, dim=-1)
        fused = ea * (1 + gamma) + beta                 # (B, hidden_dim)

        return self.head(F.mish(fused))                 # (B, context_dim)


class ActionAutoencoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(ActionAutoencoder, self).__init__()
        self.cfg = cfg

        self.encoder = nn.Sequential(
            nn.Linear(self.cfg.input_dim, self.cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.cfg.hidden_dim, self.cfg.latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.cfg.latent_dim, self.cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.cfg.hidden_dim, self.cfg.input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def train_step(self, action):
        reconstructed_action = self(action)
        loss = F.mse_loss(reconstructed_action, action)
        return loss

class ActionDeltaPredictor(nn.Module):
    # given (current condition, condition deltas) => predict action deltas
    # dimensions:
    # current condition: [batch, n_dim]
    # condition deltas: [batch, n_dim]
    # action deltas: [batch, n_channels, b_horizon]
    def __init__(self, n_dim, n_channels, b_horizon, hidden_dim=128):
        super().__init__()
        self.n_dim = n_dim
        self.n_channels = n_channels
        self.b_horizon = b_horizon
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(n_dim + b_horizon * n_channels, hidden_dim)

        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,  # Number of attention heads
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="relu",
            batch_first=True
        )

        # Transformer encoder
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, n_channels * b_horizon)
        self.output_reshape = nn.Unflatten(1, (b_horizon, n_channels))

    def forward(self, condition_deltas, current_action):
        # Flatten current_action before concatenation
        current_action = current_action.flatten(start_dim=1)
        x = torch.cat([condition_deltas, current_action], dim=1)

        x = self.input_proj(x)
        x = self.transformer(x.unsqueeze(1)).squeeze(1)
        x = self.output_proj(x)
        x = self.output_reshape(x)
        return x

    def training_step(self, condition_deltas, current_action, target_deltas):
        pred_deltas = self.forward(condition_deltas, current_action)
        loss = F.l1_loss(pred_deltas, target_deltas)
        return loss

class TransformerTracePredictor(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(TransformerTracePredictor, self).__init__()
        self.cfg = cfg

        self.encoder = nn.Sequential(
            nn.Linear(self.cfg.input_dim, self.cfg.hidden_dim),
            nn.Mish(),
            nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.cfg.hidden_dim,
                nhead=self.cfg.num_heads,
                dim_feedforward=self.cfg.hidden_dim * 4,
                dropout=self.cfg.dropout,
                activation="gelu",
            ),
            num_layers=self.cfg.num_layers,
            ),
            nn.Linear(self.cfg.hidden_dim, self.cfg.output_dim),
        )

    def forward(self, x):
        return self.encoder(x)

    def train_step(self, action, trace_label):
        predicted_trace = self(action)
        loss = F.l1_loss(predicted_trace, trace_label, reduction='none')
        loss = loss.mean(dim=1)  # Assuming the second dimension corresponds to the elements
        return loss

class CLIPContextEncoder(nn.Module):
    def __init__(self, mod1_input_dim, mod2_input_dim, modality1_encoder, modality2_encoder, embed_dim=512):
        super(CLIPContextEncoder, self).__init__()

        def create_transformer_encoder(encoder_type, input_dim):
            if encoder_type == "transformer":
                encoder = nn.Sequential(
                    nn.Linear(input_dim, embed_dim),
                    nn.Mish(),
                    nn.TransformerEncoder(
                        nn.TransformerEncoderLayer(
                        d_model=embed_dim,
                        nhead=4,  # Reduced number of attention heads
                        dim_feedforward=embed_dim * 4,
                        dropout=0.1,
                        activation="gelu",
                    ),
                    num_layers=6,  # Reduced number of layers
                    ),
                )
                encoder.output_dim = embed_dim
                return encoder
            else:
                raise ValueError(f"Unknown encoder type: {encoder_type}")

        self.modality1_encoder = create_transformer_encoder(modality1_encoder, mod1_input_dim)
        self.modality2_encoder = create_transformer_encoder(modality2_encoder, mod2_input_dim)

        self.modality1_proj = nn.Linear(self.modality1_encoder.output_dim, embed_dim)
        self.modality2_proj = nn.Linear(self.modality2_encoder.output_dim, embed_dim)

    def train_step(self, mod1, mod2):
        logits = self(mod1, mod2)
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_mod1 = F.cross_entropy(logits, labels)
        loss_mod2 = F.cross_entropy(logits.t(), labels)
        loss = (loss_mod1 + loss_mod2) / 2
        return loss

    def encode_modality1(self, x):
        return self.modality1_proj(self.modality1_encoder(x))

    def encode_modality2(self, x):
        return self.modality2_proj(self.modality2_encoder(x))

    def forward(self, mod1, mod2):
        # Normalize embeddings
        mod1_emb = F.normalize(self.encode_modality1(mod1), dim=-1)
        mod2_emb = F.normalize(self.encode_modality2(mod2), dim=-1)
        # Similarity matrix
        logits = torch.matmul(mod1_emb, mod2_emb.t())
        return logits

class TemporalResnetBlockCond(nn.Module):
    def __init__(
        self,
        in_chn,
        out_chn,
        time_context_dim,
        norm_fn=RMSNorm,
        dropout=0.0,
        use_scale_shift=True,
    ):
        super().__init__()
        self.use_scale_shift = use_scale_shift

        self.block1 = nn.ModuleList(
            [
                nn.Conv1d(in_chn, out_chn, kernel_size=3, padding=1),
                norm_fn(out_chn),
                nn.Mish(),
                nn.Dropout(dropout),
            ]
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(out_chn, out_chn, kernel_size=3, padding=1),
            norm_fn(out_chn),
            nn.Mish(),
            nn.Dropout(dropout),
        )
        self.time_context_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(time_context_dim, out_chn * 2 if use_scale_shift else out_chn),
            Rearrange("b c -> b c 1"),
        )
        self.residual_conv = (
            nn.Conv1d(in_chn, out_chn, kernel_size=1)
            if in_chn != out_chn
            else nn.Identity()
        )

    def forward(self, x, t, context):
        """
        x: batch_size x state_chn x horizon
        t: batch_size x time_dim
        context: batch_size x context_dim
        """
        t_context = torch.cat((t, context), dim=1)

        out = x.clone()
        for layer in self.block1:
            if isinstance(layer, nn.Mish):
                if self.use_scale_shift:
                    scale, shift = self.time_context_mlp(t_context).chunk(2, dim=1)
                    out = out * (1 + scale) + shift
                else:
                    out = out + self.time_context_mlp(t_context)
            out = layer(out)

        out = self.block2(out)

        return out + self.residual_conv(x)


class TemporalUnetCond(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        dim_head=32,
        num_heads=4,
        dropout=0,
        layer_mults=[1, 2, 4], # used to be [1, 2, 4, 8]
        context_args: Union[None, DictConfig] = None,
        use_flow_model=False,
    ):
        super().__init__()
        """
        Borrowed Implementation from Diffuser
        This Unet Implementation does not have the skip connection at the highest level
        4 downs and 3 ups
        no downsample at the last down layer
        output layer is the last up layer
        """
        _encoder_type = getattr(context_args, 'encoder_type', 'transformer')
        if _encoder_type == 'metrics_only':
            self.context_encoding = MetricsOnlyContextEncoder(context_args)
        elif _encoder_type == 'film':
            self.context_encoding = FiLMContextEncoder(context_args)
        elif _encoder_type == 'concat':
            self.context_encoding = ConcatContextEncoder(context_args)
        else:  # 'transformer' (default, existing behaviour)
            self.context_encoding = TransformerContextEncoder(context_args)

        dims = [input_dim, *map(lambda m: hidden_dim * m, layer_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f"[ models/temporal ] Channel dimensions: {in_out}")

        time_dim = hidden_dim
        self.context_dim = context_args.context_dim
        t_res_block = partial(
            TemporalResnetBlockCond,
            time_context_dim=self.context_dim + time_dim,
            norm_fn=RMSNorm,
            dropout=dropout,
            use_scale_shift=True,
        )
        attn_block = partial(
            SelfAttentionBlock, dim_head=dim_head, num_heads=num_heads, dropout=dropout
        )

        if use_flow_model:
            self.time_encoding = nn.Sequential(
                LearnableFourierEmbedding(embed_dim=time_dim),
                nn.Linear(time_dim, time_dim * 4),
                nn.Mish(),
                nn.Linear(time_dim * 4, time_dim),
            )
        else:
            self.time_encoding = nn.Sequential(
                SinusoidalPosEmbedding(time_dim),
                nn.Linear(time_dim, time_dim * 4),
                nn.Mish(),
                nn.Linear(time_dim * 4, time_dim),
            )

        self.down_layers = nn.ModuleList([])
        for index, (in_dim, out_dim) in enumerate(in_out):
            self.down_layers.append(
                nn.ModuleList(
                    [
                        t_res_block(in_dim, out_dim),
                        t_res_block(out_dim, out_dim),
                        Residual(PreNorm(RMSNorm(out_dim), attn_block(out_dim))),
                        (
                            DownSample(out_dim)
                            if index < len(in_out) - 1
                            else nn.Identity()
                        ),
                    ]
                )
            )

        mid_dim = in_out[-1][-1]
        self.middle_layers = nn.ModuleList(
            [
                t_res_block(mid_dim, mid_dim),
                Residual(PreNorm(RMSNorm(mid_dim), attn_block(mid_dim))),
                t_res_block(mid_dim, mid_dim),
            ]
        )

        self.up_layers = nn.ModuleList([])
        for index, (in_dim, out_dim) in enumerate(reversed(in_out[1:])):
            self.up_layers.append(
                nn.ModuleList(
                    [
                        t_res_block(out_dim * 2, in_dim),
                        t_res_block(in_dim, in_dim),
                        Residual(
                            PreNorm(
                                RMSNorm(in_dim), attn_block(in_dim, dropout=dropout)
                            )
                        ),
                        (
                            UpSample(in_dim)
                            if index < len(in_out) - 1
                            else nn.Identity()
                        ),  # if condition will always be true
                    ]
                )
            )

        in_dim, out_dim = in_out[0]
        self.output_layer = nn.Sequential(
            nn.Conv1d(out_dim, out_dim, kernel_size=3, padding=1),
            RMSNorm(out_dim),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Conv1d(out_dim, in_dim, kernel_size=1),
        )

    def forward(self, x, t, context_cond, cfg_mask):
        """
        x: batch_size x horizon x state_chn
        t: batch_size
        """
        x = rearrange(x, "b h c -> b c h")
        # Determine if we need to pad the h dimension to the next power of 2
        current_h = x.shape[2]

        next_power_of_2 = 2 ** (current_h - 1).bit_length()
        if current_h != next_power_of_2:
            # Calculate the padding needed for both sides (left and right)
            pad_size = next_power_of_2 - current_h
            # Apply padding to reach the next power of 2
            x = F.pad(x, (0, pad_size))

        t = self.time_encoding(t)

        context = self.context_encoding(context_cond)
        context[cfg_mask == 0] = 0

        h = []
        for block1, block2, attention, downsample in self.down_layers:
            x = block1(x, t, context)
            x = block2(x, t, context)
            x = attention(x)
            h.append(x)
            x = downsample(x)
        x = self.middle_layers[0](x, t, context)
        x = self.middle_layers[1](x)
        x = self.middle_layers[2](x, t, context)

        for block1, block2, attention, upsample in self.up_layers:
            x = block1(torch.cat((x, h.pop()), dim=1), t, context)
            x = block2(x, t, context)
            x = attention(x)
            x = upsample(x)

        x = self.output_layer(x)
        # If we padded earlier, remove the padding now to return to original size
        if current_h != next_power_of_2:
            # Slice to get back to the original size
            x = x[:, :, :current_h]
        return rearrange(x, "b c h -> b h c")

class MLPBlockCond(nn.Module):
    def __init__(
        self,
        in_chn,
        out_chn,
        time_context_dim,
        norm_fn=RMSNorm,
        dropout=0.0,
        use_scale_shift=True,
    ):
        super().__init__()
        self.use_scale_shift = use_scale_shift

        self.block1 = nn.ModuleList(
            [
            nn.Linear(in_chn, out_chn),
            Rearrange("b c -> b c 1"),  # Reshape to match norm_fn's expected input
            norm_fn(out_chn),
            Rearrange("b c 1 -> b c"),  # Restore original shape after normalization
            nn.Mish(),
            nn.Dropout(dropout),
            ]
        )
        self.block2 = nn.Sequential(
            nn.Linear(out_chn, out_chn),
            Rearrange("b c -> b c 1"),  # Reshape to match norm_fn's expected input
            norm_fn(out_chn),
            Rearrange("b c 1 -> b c"),  # Restore original shape after normalization
            nn.Mish(),
            nn.Dropout(dropout),
        )
        self.time_context_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(time_context_dim, out_chn * 2 if use_scale_shift else out_chn),
        )
        self.residual_conv = (
            nn.Linear(in_chn, out_chn)
            if in_chn != out_chn
            else nn.Identity()
        )

    def forward(self, x, t, context):
        """
        x: batch_size x state_chn x horizon
        t: batch_size x time_dim
        context: batch_size x context_dim
        """
        t_context = torch.cat((t, context), dim=1)

        out = x.clone()
        for layer in self.block1:
            if isinstance(layer, nn.Mish):
                if self.use_scale_shift:
                    scale, shift = self.time_context_mlp(t_context).chunk(2, dim=1)
                    out = out * (1 + scale) + shift
                else:
                    out = out + self.time_context_mlp(t_context)
            out = layer(out)

        out = self.block2(out)

        return out + self.residual_conv(x)


class MLPCond(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        dropout=0,
        layer_mults=[1, 2, 4], # used to be [1, 2, 4, 8]
        context_args: Union[None, DictConfig] = None,
    ):
        super().__init__()
        """
        Borrowed Implementation from Diffuser
        This Unet Implementation does not have the skip connection at the highest level
        4 downs and 3 ups
        no downsample at the last down layer
        output layer is the last up layer
        """
        self.context_encoding = ContextEncoder(context_args)

        dims = [input_dim, *map(lambda m: hidden_dim * m, layer_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f"[ models/temporal ] Channel dimensions: {in_out}")

        time_dim = hidden_dim
        self.context_dim = context_args.context_dim
        t_mlp_block = partial(
            MLPBlockCond,
            time_context_dim=self.context_dim + time_dim,
            norm_fn=RMSNorm,
            dropout=dropout,
            use_scale_shift=True,
        )

        self.time_encoding = nn.Sequential(
            SinusoidalPosEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.Mish(),
            nn.Linear(time_dim * 4, time_dim),
        )

        self.down_layers = nn.ModuleList([])
        for index, (in_dim, out_dim) in enumerate(in_out):
            self.down_layers.append(
                nn.ModuleList(
                    [
                        t_mlp_block(in_dim, out_dim),
                        t_mlp_block(out_dim, out_dim),
                    ]
                )
            )

        mid_dim = in_out[-1][-1]
        self.middle_layers = nn.ModuleList(
            [
                t_mlp_block(mid_dim, mid_dim),
                t_mlp_block(mid_dim, mid_dim),
            ]
        )

        self.up_layers = nn.ModuleList([])
        for index, (in_dim, out_dim) in enumerate(reversed(in_out[1:])):
            self.up_layers.append(
                nn.ModuleList(
                    [
                        t_mlp_block(out_dim, in_dim),
                        t_mlp_block(in_dim, in_dim),
                    ]
                )
            )

        in_dim, out_dim = in_out[0]
        self.output_layer = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            Rearrange("b c -> b c 1"),  # Reshape to match norm_fn's expected input
            RMSNorm(out_dim),
            Rearrange("b c 1 -> b c"),  # Restore original shape after normalization
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, in_dim),
        )

    def forward(self, x, t, context_cond, cfg_mask):
        """
        x: batch_size x horizon x state_chn
        t: batch_size
        """
        t = self.time_encoding(t)

        context = self.context_encoding(context_cond)
        context[cfg_mask == 0] = 0

        for block1, block2 in self.down_layers:
            x = block1(x, t, context)
            x = block2(x, t, context)

        x = self.middle_layers[0](x, t, context)
        x = self.middle_layers[1](x, t, context)

        for block1, block2 in self.up_layers:
            x = block1(x, t, context)
            x = block2(x, t, context)

        x = self.output_layer(x)
        return x
