"""
SDR-TR v4: TRPL + Temporal-Difference Error Context

v4 adds a TemporalDiffEncoder that captures recent-change patterns from the
input window as a lightweight "error memory" proxy (VARNN-inspired).  The
error context is concatenated with the frequency-aware state before the MLP
residual head, giving the branch information about *what changed recently* —
a strong prior for where the base forecaster is likely to err.

Changes from v3.2:
  + TemporalDiffEncoder: multi-scale Conv1D over first differences
  + state_aug = Concat([state, error_ctx])
  + MLP input dim expanded accordingly
  = r_net, hard projection, r_reg unchanged
"""

import math
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.XLinear import Forcast_multi, Forcast_with_exogenous


# ============================================================================
# Config helper
# ============================================================================

def _cfg(configs, new_name: str, default, legacy_name: Optional[str] = None):
    if hasattr(configs, new_name):
        return getattr(configs, new_name)
    if legacy_name is not None and hasattr(configs, legacy_name):
        return getattr(configs, legacy_name)
    return default


# ============================================================================
# ShapeRiskGate (unchanged from v2)
# ============================================================================

class ShapeRiskGate(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        self.use_quantile = int(_cfg(configs, 'sdr_risk_use_quantile', 1, 'hsd_risk_use_quantile'))
        self.quantile = float(_cfg(configs, 'sdr_risk_quantile', 0.70, 'hsd_risk_quantile'))
        self.threshold = float(_cfg(configs, 'sdr_risk_threshold', 1.00, 'hsd_risk_threshold'))
        self.sharpness = float(_cfg(configs, 'sdr_risk_sharpness', 8.0, 'hsd_risk_sharpness'))
        self.floor = float(_cfg(configs, 'sdr_risk_floor', 0.0, 'hsd_risk_floor'))
        self.detach_risk = int(_cfg(configs, 'sdr_detach_risk', 1, 'hsd_detach_risk'))

    def forward(self, x: torch.Tensor, out_channels: int) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, _ = x.shape
        mid = max(1, L // 2)
        q = max(1, L // 4)
        last_dev = x[:, -1, :].abs().mean(dim=-1)
        half_shift = (x[:, mid:, :].mean(dim=1) - x[:, :mid, :].mean(dim=1)).abs().mean(dim=-1)
        range_stat = (x.max(dim=1).values - x.min(dim=1).values).mean(dim=-1)
        if 2 * q <= L:
            early_slope = x[:, q:2*q, :].mean(dim=1) - x[:, :q, :].mean(dim=1)
            late_slope = x[:, -q:, :].mean(dim=1) - x[:, -2*q:-q, :].mean(dim=1)
        else:
            early_slope = x[:, -1, :] - x[:, 0, :]
            late_slope = early_slope
        accel = (late_slope - early_slope).abs().mean(dim=-1)
        score = 0.30 * last_dev + 0.35 * half_shift + 0.20 * accel + 0.15 * (range_stat / 4.0)
        if self.detach_risk:
            score = score.detach()
        if self.use_quantile and score.numel() > 1:
            thr = torch.quantile(score, q=max(0.0, min(1.0, self.quantile))).detach()
        else:
            thr = torch.tensor(self.threshold, device=x.device, dtype=x.dtype)
        risk = torch.sigmoid((score - thr) * self.sharpness).view(B, 1)
        if self.floor > 0:
            risk = self.floor + (1.0 - self.floor) * risk
        risk_h = risk.view(B, 1, 1).expand(B, self.pred_len, out_channels)
        return risk_h, risk


# ============================================================================
# FreqStateEncoder (unchanged from v3)
# ============================================================================

class FreqStateEncoder(nn.Module):
    def __init__(self, seq_len: int, channels: int, state_dim: int, dropout: float = 0.0):
        super().__init__()
        self.seq_len = seq_len
        self.channels = channels
        conv_hidden = 64
        self.conv = nn.Sequential(
            nn.Conv1d(channels, conv_hidden, kernel_size=7, padding=3),
            nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(conv_hidden, conv_hidden, kernel_size=5, padding=2),
            nn.GELU(), nn.AdaptiveAvgPool1d(1),
        )
        self.n_freq_bands = 3
        self.freq_stats_per_band = 3
        freq_feat_dim = self.n_freq_bands * self.freq_stats_per_band * channels
        self.freq_proj = nn.Linear(freq_feat_dim, conv_hidden)
        self.stat_proj = nn.Linear(6 * channels, conv_hidden)
        self.fusion = nn.Sequential(
            nn.Linear(3 * conv_hidden, state_dim),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(state_dim, state_dim),
        )

    def _spectral_features(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        x_fft = torch.fft.rfft(x, dim=1)
        mag = x_fft.abs()
        n_freq = mag.shape[1]
        band_size = max(1, n_freq // self.n_freq_bands)
        band_features = []
        for b_idx in range(self.n_freq_bands):
            start = b_idx * band_size
            end = start + band_size if b_idx < self.n_freq_bands - 1 else n_freq
            band_mag = mag[:, start:end, :]
            mean_mag = band_mag.mean(dim=1)
            freqs = torch.arange(start, end, device=x.device, dtype=x.dtype).view(1, -1, 1)
            centroid = (band_mag * freqs).sum(dim=1) / band_mag.sum(dim=1).clamp(min=1e-8)
            band_sum = band_mag.sum(dim=1, keepdim=True).clamp(min=1e-8)
            band_norm = band_mag / band_sum
            entropy = -(band_norm * (band_norm + 1e-8).log()).sum(dim=1)
            entropy = entropy / max(1.0, math.log(end - start + 1))
            band_features.append(torch.cat([mean_mag, centroid, entropy], dim=-1))
        return torch.cat(band_features, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        conv_feat = self.conv(x.permute(0, 2, 1)).squeeze(-1)
        freq_feat = self.freq_proj(self._spectral_features(x))
        mid = max(1, L // 2)
        mean = x.mean(dim=1)
        std = torch.sqrt(torch.var(x, dim=1, unbiased=False) + 1e-5)
        trend = x[:, -1, :] - x[:, 0, :]
        last = x[:, -1, :]
        half_shift = x[:, mid:, :].mean(dim=1) - x[:, :mid, :].mean(dim=1)
        range_stat = x.max(dim=1).values - x.min(dim=1).values
        stats = torch.cat([mean, std, trend, last, half_shift, range_stat], dim=-1)
        stat_feat = self.stat_proj(stats)
        return self.fusion(torch.cat([conv_feat, freq_feat, stat_feat], dim=-1))


# ============================================================================
# TemporalDiffEncoder — v4 error-context encoder  ← NEW
# ============================================================================

class TemporalDiffEncoder(nn.Module):
    """Multi-scale temporal-difference encoder (VARNN-inspired error proxy).

    Computes first differences at the input window and runs a lightweight Conv1D
    over them to capture "what changed recently" — a strong prior for where the
    base forecaster is likely to make errors.
    """

    def __init__(self, channels: int, error_dim: int = 32, dropout: float = 0.0):
        super().__init__()
        self.error_dim = error_dim
        self.conv = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(32, error_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, C] -> error_ctx: [B, error_dim]."""
        # First differences along time axis
        diff = x[:, 1:, :] - x[:, :-1, :]         # [B, L-1, C]
        diff = F.pad(diff.permute(0, 2, 1), (1, 0))  # [B, C, L], pad start
        feat = self.conv(diff).squeeze(-1)          # [B, 32]
        return self.proj(feat)                      # [B, error_dim]


# ============================================================================
# TRPLResidualBranch — v4 with error context
# ============================================================================

class TRPLResidualBranch(nn.Module):
    """v4: TRPL-based trust-region residual branch + temporal-diff error context.

    state_enc = FreqStateEncoder(X)              # frequency-aware state
    error_ctx = TemporalDiffEncoder(X)           # recent-change patterns
    Δ_raw = MLP([state_enc, error_ctx])          # augmented residual head
    r_h = r_min + (r_max-r_min)*sigmoid(r_net(state_enc, h_emb))
    Δ_TR = hard_project(Δ_raw, r_h)
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.out_channels = self.channels if configs.features == 'M' else 1

        self.state_dim = int(_cfg(configs, 'sdr_state_dim', 128, 'hsd_state_dim'))
        self.dropout_p = float(_cfg(configs, 'sdr_dropout', 0.0, 'hsd_dropout'))
        self.residual_scale = max(
            float(_cfg(configs, 'sdr_residual_scale', 0.15, 'hsd_residual_scale')), 1e-6
        )
        self.r_min = float(_cfg(configs, 'sdr_tr_r_min', 0.005))
        self.r_max = float(_cfg(configs, 'sdr_tr_r_max', 0.15))
        self.r_reg_lambda = float(_cfg(configs, 'sdr_tr_r_reg', 0.001))

        # v4: error context dimension
        self.error_dim = int(_cfg(configs, 'sdr_error_dim', 32))

        # State encoder (frequency-aware, unchanged)
        self.state_encoder = FreqStateEncoder(
            seq_len=self.seq_len, channels=self.channels,
            state_dim=self.state_dim, dropout=self.dropout_p,
        )

        # v4: temporal-diff error encoder
        self.error_encoder = TemporalDiffEncoder(
            channels=self.channels, error_dim=self.error_dim, dropout=self.dropout_p,
        )

        # Risk gate
        self.risk_gate = ShapeRiskGate(configs)

        # MLP: [state + error_ctx] → raw residual
        mlp_in = self.state_dim + self.error_dim
        out_features = self.out_channels * self.pred_len
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, self.state_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.state_dim, out_features),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        # Horizon embedding
        self.horizon_emb = nn.Parameter(torch.randn(1, self.pred_len, 8) * 0.02)

        # r_net: state → per-horizon trust radius
        r_hidden = max(32, self.state_dim // 2)
        self.r_net = nn.Sequential(
            nn.Linear(self.state_dim + 8, r_hidden),
            nn.GELU(),
            nn.Dropout(self.dropout_p * 0.5),
            nn.Linear(r_hidden, 1),
        )
        nn.init.zeros_(self.r_net[-1].weight)
        nn.init.constant_(self.r_net[-1].bias, -4.0)  # init r≈0.008

        self.last_r_mean: float = 0.0
        self.last_debug: Dict[str, float] = {}

    def forward(self, x: torch.Tensor):
        B, H = x.shape[0], self.pred_len

        # 1. Frequency-aware state
        state = self.state_encoder(x)  # [B, state_dim]

        # 2. v4: Temporal-diff error context
        error_ctx = self.error_encoder(x)  # [B, error_dim]

        # 3. Augmented state → raw residual
        state_aug = torch.cat([state, error_ctx], dim=-1)  # [B, state_dim + error_dim]
        raw = self.mlp(state_aug)
        raw_delta = raw.view(B, self.out_channels, H).permute(0, 2, 1)  # [B, H, Cout]

        # 4. Per-horizon trust radius
        state_exp = state.unsqueeze(1).expand(B, H, self.state_dim)
        h_emb = self.horizon_emb.expand(B, H, 8)
        r_input = torch.cat([state_exp, h_emb], dim=-1)
        r_raw = self.r_net(r_input).squeeze(-1)
        r_h = self.r_min + (self.r_max - self.r_min) * torch.sigmoid(r_raw)

        # 5. Risk modulation
        risk, risk_scalar = self.risk_gate(x, out_channels=self.out_channels)
        risk_h = risk.mean(dim=-1)
        r_effective = r_h * (0.8 + 0.4 * risk_h)
        r_effective = torch.clamp(r_effective, min=self.r_min, max=self.r_max)

        # 6. Hard TRPL projection
        raw_norm = raw_delta.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
        r_expanded = r_effective.unsqueeze(-1)
        scale = torch.clamp(r_expanded / raw_norm, max=1.0)
        correction = scale * raw_delta

        # 7. Soft clipping
        correction = torch.tanh(correction / self.residual_scale) * self.residual_scale

        if self.training:
            with torch.no_grad():
                self.last_debug = {
                    'r_mean': float(r_effective.mean().detach().cpu()),
                    'r_std': float(r_effective.std().detach().cpu()),
                    'proj_activated': float((scale < 1.0).float().mean().detach().cpu()),
                    'correction_abs_mean': float(correction.abs().mean().detach().cpu()),
                }

        self.last_r_mean = float(r_effective.mean().detach().cpu())
        gate = r_expanded.expand(B, H, self.out_channels)
        return correction, raw_delta, gate, risk


# ============================================================================
# Model
# ============================================================================

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
        self.disable_dep = int(_cfg(configs, 'sdr_disable_dep', 0, 'hsd_disable_dep'))
        self.hsd_debug = int(_cfg(configs, 'sdr_debug', 0, 'hsd_debug'))
        self._debug_counter = 0

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
            print('[SDR-TR v4] dependency branch DISABLED — XLinear-only mode')
        else:
            self.dep_branch = TRPLResidualBranch(configs)
            print('[SDR-TR v4] TRPLResidualBranch + TemporalDiffEncoder active')
            print(f'  r_range=[{self.dep_branch.r_min:.4f}, {self.dep_branch.r_max:.4f}]')
            print(f'  state_dim={self.dep_branch.state_dim} error_dim={self.dep_branch.error_dim}')

        self._last_means = None
        self._last_stdev = None
        self._last_y_self_norm = None

    def _self_forecast(self, x_norm: torch.Tensor) -> torch.Tensor:
        x_ch = x_norm.permute(0, 2, 1)
        en = self.self_backbone(x_ch)
        return self.self_head(en).permute(0, 2, 1).contiguous()

    def _forecast_normed(self, x_norm: torch.Tensor) -> torch.Tensor:
        y_self = self._self_forecast(x_norm)
        self._last_y_self_norm = y_self
        if self.disable_dep:
            return y_self
        correction, _, _, _ = self.dep_branch(x_norm)
        y = y_self + correction
        if self.hsd_debug and self.training:
            if self._debug_counter == 0 or self._debug_counter % 100 == 0:
                dbg = getattr(self.dep_branch, 'last_debug', {})
                print('[SDR-TR v4 DEBUG] ' + ', '.join([f'{k}={v:.6f}' for k, v in dbg.items()]))
            self._debug_counter += 1
        return y

    def get_aux_loss(self, target=None):
        """v4: r_reg penalty only."""
        if self.disable_dep or self.dep_branch is None:
            return None
        r_mean = getattr(self.dep_branch, 'last_r_mean', 0.0)
        if r_mean > 0:
            return self.dep_branch.r_reg_lambda * r_mean
        return None

    def get_debug_stats(self):
        if self.disable_dep or self.dep_branch is None:
            return {}
        return dict(getattr(self.dep_branch, 'last_debug', {}))

    def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        if self.norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x = x_enc - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = x / stdev
        else:
            x = x_enc
            means = None
            stdev = None
        self._last_means = means
        self._last_stdev = stdev
        y_norm = self._forecast_normed(x)
        if self.norm:
            if self.feature == 'M':
                y = y_norm * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
                y = y + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            else:
                y = y_norm * stdev[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1)
                y = y + means[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1)
        else:
            y = y_norm
        return y
