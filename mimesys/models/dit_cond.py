"""DiT-style 2D Transformer denoiser for the (S=20 stressors, T=20 threads)
action tensor.

Each (s, t) cell is one token. Two learnable positional embeddings (stressor_id
and thread_id) added per token. AdaLN-zero modulation by (time + context) at
every block. Final unpatch back to (B, S, T).

Unlike the U-Net, makes no spatial-locality assumption on the categorical
stressor axis.
"""
from __future__ import annotations

import os
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from mimesys.models.utils import SinusoidalPosEmbedding, LearnableFourierEmbedding
from mimesys.models.temporal_unet_cond import (
    MetricsOnlyContextEncoder,
    TransformerMetricsOnlyEncoder,
    TransformerContextEncoder,
    TransformerFullContextEncoder,
    PerPositionFiLMEncoder,
    ConcatContextEncoder,
    PrevSumsContextEncoder,
)


def _build_encoder(context_args):
    enc_type = getattr(context_args, "encoder_type", "metrics_only")
    if enc_type == "metrics_only":
        return MetricsOnlyContextEncoder(context_args), False
    if enc_type == "transformer_metrics_only":
        return TransformerMetricsOnlyEncoder(context_args), False
    if enc_type == "transformer":
        return TransformerContextEncoder(context_args), False
    if enc_type == "transformer_full":
        return TransformerFullContextEncoder(context_args), False
    if enc_type == "film":
        return PerPositionFiLMEncoder(context_args), True   # returns per-thread
    if enc_type == "concat":
        return ConcatContextEncoder(context_args), False
    if enc_type == "prev_sums":
        return PrevSumsContextEncoder(context_args), False
    raise ValueError(f"DiTCond: unsupported encoder_type {enc_type}")


class DiTBlock(nn.Module):
    """Single DiT block: AdaLN(scale,shift,gate) before self-attn, then
    AdaLN(scale,shift,gate) before MLP. Conditioning -> 6 modulation
    parameters per block. Accepts either:
      - global cond (B, cond_dim)            → same modulation per token
      - per-token cond (B, N, cond_dim)      → token-wise modulation
    """
    def __init__(self, d_model: int, num_heads: int, mlp_ratio: int = 4,
                 dropout: float = 0.0, cond_dim: int = 256):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attn  = nn.MultiheadAttention(d_model, num_heads,
                                            dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.mlp   = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * mlp_ratio, d_model),
            nn.Dropout(dropout),
        )
        self.adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * d_model),
        )
        # Zero-init the AdaLN-zero output so each block starts as identity.
        nn.init.zeros_(self.adaln[-1].weight)
        nn.init.zeros_(self.adaln[-1].bias)

    def forward(self, x, cond):
        """x: (B, N, d). cond: (B, cond_dim) or (B, N, cond_dim)."""
        proj = self.adaln(cond)
        s1, h1, g1, s2, h2, g2 = proj.chunk(6, dim=-1)
        if proj.dim() == 2:
            s1, h1, g1, s2, h2, g2 = [v.unsqueeze(1) for v in (s1, h1, g1, s2, h2, g2)]
        y  = self.norm1(x) * (1 + s1) + h1
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + g1 * y
        y = self.norm2(x) * (1 + s2) + h2
        y = self.mlp(y)
        x = x + g2 * y
        return x


class DiTFinal(nn.Module):
    """Final AdaLN + unpatch to scalar per (s, t) cell."""
    def __init__(self, d_model: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * d_model),
        )
        nn.init.zeros_(self.adaln[-1].weight)
        nn.init.zeros_(self.adaln[-1].bias)
        self.proj = nn.Linear(d_model, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, cond):
        proj = self.adaln(cond)
        s, h = proj.chunk(2, dim=-1)
        if proj.dim() == 2:
            s, h = s.unsqueeze(1), h.unsqueeze(1)
        y = self.norm(x) * (1 + s) + h
        return self.proj(y).squeeze(-1)   # (B, N)


class DiTCond(nn.Module):
    """2D-positional DiT denoiser.

    Cfg expectations (all read from context_args):
      encoder_type, input_dim (=metric dim), hidden_dim, context_dim,
      dropout, num_heads, num_layers, action_dim (only for encoders that use it).

    Extra knobs (read from env):
      MIMESYS_DIT_DMODEL  (default 256)
      MIMESYS_DIT_LAYERS  (default 6)
      MIMESYS_DIT_HEADS   (default 8)
    """
    def __init__(
        self,
        num_stressors: int = 20,
        num_threads: int = 20,
        context_args: Union[None, DictConfig] = None,
        **_ignored,
    ):
        super().__init__()
        d_model    = int(os.environ.get("MIMESYS_DIT_DMODEL", "256"))
        num_layers = int(os.environ.get("MIMESYS_DIT_LAYERS", "6"))
        num_heads  = int(os.environ.get("MIMESYS_DIT_HEADS", "8"))
        mlp_ratio  = 4
        dropout    = float(getattr(context_args, "dropout", 0.1))
        # Per-thread conditioning mode. 'none' (default) — same global cond per
        # token. 'bias' — per-core CPU% added as additive token bias. 'film' —
        # per-position FiLM (each token gets its own AdaLN modulation derived
        # from per-core CPU). 'film_enc' — use PerPositionFiLMEncoder output
        # broadcast across stressors at each thread index.
        self.thr_cond_mode = os.environ.get("MIMESYS_DIT_THR_COND", "none")

        self.num_stressors = num_stressors
        self.num_threads   = num_threads
        self.d_model       = d_model

        # Per-cell scalar → d_model token embedding.
        self.patch_proj = nn.Linear(1, d_model)
        # 2D positional embeddings (stressor + thread).
        self.stressor_pos = nn.Embedding(num_stressors, d_model)
        self.thread_pos   = nn.Embedding(num_threads,   d_model)

        # Time embedding (sinusoidal → MLP). cond_dim = time_dim + context_dim.
        time_dim    = d_model
        context_dim = int(context_args.context_dim)
        self.time_encoding = nn.Sequential(
            SinusoidalPosEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.Mish(),
            nn.Linear(time_dim * 4, time_dim),
        )

        # Context encoder (delegated to the same factory as TemporalUnetCond).
        self.context_encoding, self._ctx_is_per_position = _build_encoder(context_args)

        # IO mask buffer, shared by the per-thread cond modes (bias, film).
        # MIMESYS_DIT_BIAS_NONIO=1 zeros the per-thread CPU% signal on
        # IO-stressor tokens (idx 12..18 by default), since per-core CPU
        # doesn't drive IO behavior.
        if os.environ.get("MIMESYS_DIT_BIAS_NONIO", "0") == "1":
            io_idx_str = os.environ.get(
                "MIMESYS_DIT_BIAS_NONIO_IO_IDX", "12,13,14,15,16,17,18")
            io_idx = [int(x) for x in io_idx_str.split(",") if x.strip()]
            mask = torch.ones(self.num_stressors, dtype=torch.float32)
            for i in io_idx:
                if 0 <= i < self.num_stressors:
                    mask[i] = 0.0
            # Buffer name kept for backward compat with existing NONIO ckpts.
            self.register_buffer("_bias_stressor_mask", mask)   # (S,)
            print(f"[dit] {self.thr_cond_mode} mode: IO mask active — applied to "
                  f"{int(mask.sum().item())}/{self.num_stressors} stressor rows "
                  f"(IO idx zeroed: {io_idx})", flush=True)
        else:
            self._bias_stressor_mask = None

        # Per-thread conditioning heads
        if self.thr_cond_mode == "bias":
            # 1 → d_model projection for the per-core CPU scalar per thread.
            self.per_core_proj = nn.Sequential(
                nn.Linear(1, d_model), nn.Mish(),
                nn.Linear(d_model, d_model),
            )
        elif self.thr_cond_mode == "film":
            # per-core CPU + thread position → cond vector per token.
            self.per_core_emb = nn.Sequential(
                nn.Linear(1, d_model), nn.Mish(),
                nn.Linear(d_model, context_dim),
            )
        elif self.thr_cond_mode == "film_enc":
            # Use PerPositionFiLMEncoder's (B, ctx, T) output as per-thread ctx.
            # Force the context encoder to be the per-position one if not already.
            if not self._ctx_is_per_position:
                from mimesys.models.temporal_unet_cond import PerPositionFiLMEncoder
                self.context_encoding = PerPositionFiLMEncoder(context_args)
                self._ctx_is_per_position = True

        # Prev-action conditioning. 'none' (default) — prev_action reaches the
        # model only if the context encoder consumes it. 'token' — each (s, t)
        # token additionally receives its own prev_action[s, t] scalar through a
        # shared Linear(1, d) head added to the token embedding. This keeps prev
        # conditioning spatially aligned with the token grid and avoids the
        # flattened-400-dim concat pathway: with mostly-(-1) sparse prev grids,
        # a global concat MLP develops a 400-wide coherent bias direction whose
        # gradient is ~400x a normal bias's, which AdaLN then amplifies
        # multiplicatively (no post-modulation norm) — unstable at lr=1e-4.
        self.prev_cond_mode = os.environ.get(
            "MIMESYS_DIT_PREV_COND",
            str(getattr(context_args, "prev_cond_mode", "none")))
        if self.prev_cond_mode == "token":
            self.prev_proj = nn.Sequential(
                nn.Linear(1, d_model), nn.Mish(),
                nn.Linear(d_model, d_model),
            )
            # MIMESYS_PREV_FREEZE=1: ablation control — keep the prev_proj
            # module but hard-wire its input to the all-(-1) "no prev" grid,
            # so the module can only act as learned per-token structure and
            # never sees real history. Separates architectural-prior gains
            # from genuine history conditioning.
            self.prev_freeze = os.environ.get("MIMESYS_PREV_FREEZE", "0") == "1"
            print(f"[dit] prev_cond_mode=token — per-token prev_action bias active"
                  f"{' (FROZEN to -1)' if self.prev_freeze else ''}",
                  flush=True)

        self.cond_dim = time_dim + context_dim
        self.blocks   = nn.ModuleList([
            DiTBlock(d_model, num_heads, mlp_ratio, dropout, self.cond_dim)
            for _ in range(num_layers)
        ])
        self.final = DiTFinal(d_model, self.cond_dim)

    def forward(self, x, t, context_cond, cfg_mask):
        """
        x: (B, S, T)  — noise-corrupted action
        t: (B,)       — timestep
        context_cond: dict
        cfg_mask: (B,) — 1=keep cond, 0=drop (for CFG)
        """
        B, S, T = x.shape
        device = x.device
        # Tokenize each (s, t) cell.
        tokens = self.patch_proj(x.reshape(B, S * T, 1))           # (B, N=S*T, d)
        s_ids  = torch.arange(S, device=device).unsqueeze(-1).expand(-1, T).reshape(-1)
        t_ids  = torch.arange(T, device=device).unsqueeze(0).expand(S, -1).reshape(-1)
        tokens = tokens + self.stressor_pos(s_ids).unsqueeze(0) + \
                          self.thread_pos(t_ids).unsqueeze(0)      # (B, N, d)

        # Per-core CPU% (first T metric dims) — used by all per-thread modes.
        metric = context_cond['metric']                             # (B, M=28)
        per_core = metric[:, :T].unsqueeze(-1)                      # (B, T, 1)

        # 'bias' mode: additive per-thread token bias.
        if self.thr_cond_mode == "bias":
            thread_bias = self.per_core_proj(per_core)              # (B, T, d)
            # Broadcast across S stressors as (B, S, T, d). Optional IO mask
            # zeros bias on IO-stressor rows (see __init__).
            thread_bias = thread_bias.unsqueeze(1).expand(-1, S, -1, -1)
            if getattr(self, "_bias_stressor_mask", None) is not None:
                # (S,) → (1, S, 1, 1) → broadcast across B, T, d
                thread_bias = thread_bias * self._bias_stressor_mask.view(1, S, 1, 1)
            thread_bias = thread_bias.reshape(B, S * T, self.d_model)
            tokens = tokens + thread_bias

        # 'token' prev conditioning: add each cell's prev weight to its token.
        # CFG-masked like the rest of the conditioning so the unconditional
        # branch stays truly unconditional.
        if self.prev_cond_mode == "token":
            prev = context_cond['prev_action']
            if getattr(self, "prev_freeze", False):
                prev = torch.full_like(prev, -1.0)
            prev = prev.reshape(B, S * T, 1)                          # (B, N, 1)
            tokens = tokens + self.prev_proj(prev) * cfg_mask.view(-1, 1, 1)

        # Time + (CFG-masked) context → cond.
        time_emb = self.time_encoding(t)                            # (B, time_dim)
        ctx = self.context_encoding(context_cond)                   # (B, ctx_dim) or (B, ctx_dim, T)

        # Per-token cond construction.
        if self.thr_cond_mode == "film":
            # Token (s, t) sees per_core_emb(metric[t]) + global_ctx. The
            # global encoder output is added in so its params receive
            # gradients (Lightning otherwise raises "unused parameters").
            if self._ctx_is_per_position:
                ctx_global = ctx.mean(dim=-1)                       # collapse to global
            else:
                ctx_global = ctx                                     # (B, context_dim)
            ctx_global = ctx_global * cfg_mask.view(-1, 1)           # CFG drop
            ctx_thread_per_core = self.per_core_emb(per_core)        # (B, T, context_dim)
            ctx_thread_per_core = ctx_thread_per_core * cfg_mask.view(-1, 1, 1)
            ctx_thread = ctx_thread_per_core + ctx_global.unsqueeze(1)   # broadcast global add
            if getattr(self, "_bias_stressor_mask", None) is not None:
                # IO mask: on IO stressor rows, use only ctx_global (no per_core).
                # ctx_4d_with: (B, S, T, ctx_dim) per-token with per_core for all rows
                ctx_4d_with = ctx_thread.unsqueeze(1).expand(-1, S, -1, -1)
                ctx_4d_only_global = (ctx_global.unsqueeze(1).unsqueeze(1)
                                      .expand(-1, S, T, -1))         # (B, S, T, ctx_dim)
                mask = self._bias_stressor_mask.view(1, S, 1, 1)
                ctx_4d = ctx_4d_with * mask + ctx_4d_only_global * (1 - mask)
                ctx_per_tok = ctx_4d.reshape(B, S * T, -1)
            else:
                ctx_per_tok = (ctx_thread.unsqueeze(1)
                                          .expand(-1, S, -1, -1)
                                          .reshape(B, S * T, -1))
            time_per_tok = time_emb.unsqueeze(1).expand(-1, S * T, -1)
            cond = torch.cat([time_per_tok, ctx_per_tok], dim=-1)   # (B, N, cond_dim)
        elif self.thr_cond_mode == "film_enc":
            # Per-thread ctx from PerPositionFiLMEncoder: (B, context_dim, T).
            ctx_thread = ctx.transpose(1, 2)                        # (B, T, ctx_dim)
            ctx_thread = ctx_thread * cfg_mask.view(-1, 1, 1)
            ctx_per_tok = (ctx_thread.unsqueeze(1)
                                      .expand(-1, S, -1, -1)
                                      .reshape(B, S * T, -1))
            time_per_tok = time_emb.unsqueeze(1).expand(-1, S * T, -1)
            cond = torch.cat([time_per_tok, ctx_per_tok], dim=-1)
        else:
            # 'none' or 'bias': global cond per sample.
            if self._ctx_is_per_position:
                ctx = ctx.mean(dim=-1)                              # collapse to global
            ctx = ctx * cfg_mask.view(-1, 1)
            cond = torch.cat([time_emb, ctx], dim=-1)               # (B, cond_dim)

        for blk in self.blocks:
            tokens = blk(tokens, cond)
        out = self.final(tokens, cond)                              # (B, N)
        return out.reshape(B, S, T)
