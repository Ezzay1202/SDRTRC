"""
HSDNet-v1.1: Horizon-aware State-conditioned Dependency Network

This version keeps XLinear's lightweight self-history branch as the protected
base predictor and adds a state-conditioned, horizon-aware dependency residual
branch.  Compared with HSDNet-v1, v1.1 is baseline-safe:

  1) With --hsd_disable_dep 1, the dependency branch is not instantiated, so the
     model is exactly the XLinear self branch.  This is for sanity checking.
  2) In normal residual mode, the dependency output layer is zero-initialized,
     so the initial prediction is exactly Y_self.  The dependency branch learns
     a correction DeltaY during training.
  3) Dependency branch initialization can be RNG-safe, so model construction does
     not shift the subsequent random trajectory relative to XLinear.

Input shape : x_enc [B, L, C]
Output shape: [B, H, C] for M, [B, H, 1] for MS/S
"""

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.XLinear import Forcast_multi, Forcast_with_exogenous


class StateEncoder(nn.Module):
    """Encode current window state from robust statistics."""

    def __init__(self, seq_len: int, channels: int, state_dim: int, dropout: float = 0.0):
        super().__init__()
        self.seq_len = seq_len
        self.channels = channels
        in_dim = 4 * channels  # mean, std, trend, last value for each channel
        hidden = max(state_dim, min(4 * state_dim, 256))
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, state_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, C] -> state: [B, state_dim]."""
        mean = x.mean(dim=1)
        std = torch.sqrt(torch.var(x, dim=1, unbiased=False) + 1e-5)
        trend = x[:, -1, :] - x[:, 0, :]
        last = x[:, -1, :]
        stats = torch.cat([mean, std, trend, last], dim=-1)
        return self.net(stats)


class HorizonStateDependencyBranch(nn.Module):
    """State-conditioned, horizon-aware dependency residual branch.

    The branch obtains variable-wise embeddings from channel histories. For each
    horizon group, it builds a state-conditioned dependency matrix A_g(target,
    source), then aggregates source residual forecasts into target residuals.
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.feature = configs.features
        self.out_channels = self.channels if self.feature == 'M' else 1

        self.dep_dim = int(getattr(configs, 'hsd_dep_dim', 16))
        self.state_dim = int(getattr(configs, 'hsd_state_dim', 32))
        self.num_groups = max(1, int(getattr(configs, 'hsd_num_groups', 4)))
        self.num_groups = min(self.num_groups, self.pred_len)
        self.topk = int(getattr(configs, 'hsd_topk', 0))
        self.dropout_p = float(getattr(configs, 'hsd_dropout', 0.0))
        self.gate_init = float(getattr(configs, 'hsd_gate_init', -3.0))
        self.temperature = max(float(getattr(configs, 'hsd_temperature', 1.0)), 1e-6)
        self.zero_init = int(getattr(configs, 'hsd_zero_init', 1))

        self.state_encoder = StateEncoder(
            seq_len=self.seq_len,
            channels=self.channels,
            state_dim=self.state_dim,
            dropout=self.dropout_p,
        )

        self.value_proj = nn.Sequential(
            nn.Linear(self.seq_len, self.dep_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_p),
        )

        self.key_proj = nn.Linear(self.dep_dim, self.dep_dim)
        self.query_proj = nn.Linear(self.dep_dim, self.dep_dim)
        self.state_to_query = nn.Linear(self.state_dim, self.num_groups * self.dep_dim)
        self.group_embedding = nn.Parameter(torch.randn(self.num_groups, self.dep_dim) * 0.02)
        self.query_norm = nn.LayerNorm(self.dep_dim)
        self.key_norm = nn.LayerNorm(self.dep_dim)

        # Shared source residual forecast. It maps each variable embedding to a
        # horizon residual.  This layer is zero-initialized in v1.1 so normal
        # HSDNet starts exactly from the XLinear self branch.
        self.source_dropout = nn.Dropout(self.dropout_p)
        self.source_linear = nn.Linear(self.dep_dim, self.pred_len)
        if self.zero_init:
            nn.init.zeros_(self.source_linear.weight)
            nn.init.zeros_(self.source_linear.bias)

        gate_hidden = max(self.state_dim, 16)
        self.gate_head = nn.Sequential(
            nn.Linear(self.state_dim, gate_hidden),
            nn.GELU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(gate_hidden, self.num_groups),
        )
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.constant_(self.gate_head[-1].bias, self.gate_init)

        self._groups = self._make_horizon_groups(self.pred_len, self.num_groups)

    @staticmethod
    def _make_horizon_groups(pred_len: int, num_groups: int) -> List[Tuple[int, int]]:
        groups = []
        for g in range(num_groups):
            start = (g * pred_len) // num_groups
            end = ((g + 1) * pred_len) // num_groups
            if end > start:
                groups.append((start, end))
        return groups

    def _apply_topk(self, scores: torch.Tensor) -> torch.Tensor:
        """Keep top-k source variables for each target and horizon group.

        scores: [B, G, T, C]
        """
        if self.topk <= 0 or self.topk >= scores.shape[-1]:
            return scores
        vals, idx = torch.topk(scores, k=self.topk, dim=-1)
        masked = torch.full_like(scores, float('-inf'))
        masked.scatter_(-1, idx, vals)
        return masked

    def forward(self, x: torch.Tensor):
        """Return dependency residual and fusion gate.

        Args:
            x: normalized input [B, L, C]
        Returns:
            y_delta: [B, H, C] for M or [B, H, 1] for MS/S
            gate   : [B, H, C] for M or [B, H, 1] for MS/S
        """
        b, _, c = x.shape
        assert c == self.channels, f"Expected {self.channels} channels, got {c}"

        x_ch = x.permute(0, 2, 1)  # [B, C, L]
        z = self.value_proj(x_ch)  # [B, C, d]

        state = self.state_encoder(x)           # [B, state_dim]
        keys = self.key_norm(self.key_proj(z))  # [B, C, d]
        base_queries = self.query_proj(z)       # [B, C, d]

        if self.feature == 'M':
            target_queries = base_queries       # [B, C, d]
        else:
            target_queries = base_queries[:, -1:, :]  # [B, 1, d]

        state_q = self.state_to_query(state).view(b, self.num_groups, self.dep_dim)
        q = target_queries.unsqueeze(1) + state_q.unsqueeze(2) + self.group_embedding.view(
            1, self.num_groups, 1, self.dep_dim
        )
        q = self.query_norm(q)

        scale = math.sqrt(self.dep_dim) * self.temperature
        scores = torch.einsum('bgtd,bcd->bgtc', q, keys) / scale  # [B, G, T, C]
        scores = self._apply_topk(scores)
        attn = torch.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout_p, training=self.training)

        src_delta = self.source_linear(self.source_dropout(z))  # [B, C, H]

        delta_chunks = []
        gate_chunks = []
        group_gate = torch.sigmoid(self.gate_head(state))  # [B, G]
        for g, (start, end) in enumerate(self._groups):
            src_g = src_delta[:, :, start:end]             # [B, C, Hg]
            attn_g = attn[:, g, :, :]                      # [B, T, C]
            delta_g = torch.einsum('btc,bch->bth', attn_g, src_g)  # [B, T, Hg]
            delta_chunks.append(delta_g)

            gate_g = group_gate[:, g].view(b, 1, 1).expand(b, self.out_channels, end - start)
            gate_chunks.append(gate_g)

        y_delta = torch.cat(delta_chunks, dim=-1).permute(0, 2, 1).contiguous()
        gate = torch.cat(gate_chunks, dim=-1).permute(0, 2, 1).contiguous()
        return y_delta, gate


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.channel = configs.enc_in
        self.t_ff = configs.t_ff
        self.c_ff = configs.c_ff
        self.norm = configs.usenorm
        self.embed_dropout = configs.embed_dropout
        self.head_dropout = configs.head_dropout
        self.t_dropout = configs.t_dropout
        self.c_dropout = configs.c_dropout
        self.feature = configs.features
        self.fusion = str(getattr(configs, 'hsd_fusion', 'residual')).lower()
        self.disable_dep = int(getattr(configs, 'hsd_disable_dep', 0))
        self.rng_safe_init = int(getattr(configs, 'hsd_rng_safe_init', 1))

        if self.feature == 'M':
            self.self_backbone = Forcast_multi(
                self.seq_len, self.d_model, self.channel, self.t_ff,
                self.c_ff, self.t_dropout, self.c_dropout, self.embed_dropout
            )
        else:
            self.self_backbone = Forcast_with_exogenous(
                self.seq_len, self.d_model, self.channel, self.t_ff,
                self.c_ff, self.t_dropout, self.c_dropout, self.embed_dropout
            )

        self.self_head = nn.Sequential(
            nn.Dropout(self.head_dropout),
            nn.Linear(2 * self.d_model, self.pred_len)
        )

        if self.disable_dep:
            self.dep_branch = None
            print('[HSDNet-v1.1] dependency branch is NOT instantiated. '
                  'This run should be equivalent to the XLinear self branch.')
        else:
            if self.rng_safe_init:
                cpu_rng_state = torch.get_rng_state()
                cuda_rng_state = None
                if torch.cuda.is_available():
                    try:
                        cuda_rng_state = torch.cuda.get_rng_state_all()
                    except Exception:
                        cuda_rng_state = None
                self.dep_branch = HorizonStateDependencyBranch(configs)
                torch.set_rng_state(cpu_rng_state)
                if cuda_rng_state is not None:
                    try:
                        torch.cuda.set_rng_state_all(cuda_rng_state)
                    except Exception:
                        pass
            else:
                self.dep_branch = HorizonStateDependencyBranch(configs)

    def _self_forecast(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Run the original XLinear branch on normalized data."""
        x_ch = x_norm.permute(0, 2, 1)  # [B, C, L]
        en = self.self_backbone(x_ch)
        y_self = self.self_head(en).permute(0, 2, 1).contiguous()
        return y_self

    def _forecast_normed(self, x_norm: torch.Tensor) -> torch.Tensor:
        y_self = self._self_forecast(x_norm)

        if self.disable_dep:
            return y_self

        y_delta, gate = self.dep_branch(x_norm)

        if self.fusion in ['interp', 'interpolate']:
            # Kept only for ablation/backward compatibility.  Not recommended
            # as the default because it is not strictly baseline-safe unless
            # y_delta is interpreted as a full alternative prediction.
            y = y_self + gate * (y_delta - y_self)
        else:
            # Default v1.1 fusion: residual correction.  With zero-init source
            # head, y_delta=0 at initialization and output equals y_self.
            y = y_self + gate * y_delta
        return y

    def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        """x_enc: [B, L, C]."""
        if self.norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x = x_enc - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = x / stdev
        else:
            x = x_enc
            means = None
            stdev = None

        y = self._forecast_normed(x)

        if self.norm:
            if self.feature == 'M':
                y = y * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
                y = y + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            else:
                y = y * stdev[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1)
                y = y + means[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1)
        return y
