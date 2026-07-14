"""
HSDNet-v3: Residual-Targeted Horizon-aware State-conditioned Dependency Network

This version keeps XLinear as a protected self-history branch and trains the
state-conditioned dependency branch with an explicit residual target:

    residual_target = stopgrad(Y_true_norm - Y_self_norm)

The final prediction is still conservative:

    Y = Y_self + risk(X) * gate(X,h) * bounded_delta(X,h)

but the dependency branch no longer relies only on weak end-to-end gradients.
It receives direct supervision to model the part that the XLinear self branch
misses.  This is intended to avoid the two failure modes observed before:

- v1/v1.1: an unconstrained residual branch damaged the strong XLinear branch;
- v2: a strongly bounded branch stayed almost zero and behaved like XLinear.

Key properties
--------------
1) Strict XLinear sanity check:
   --hsd_disable_dep 1 => dependency branch is not instantiated, and HSDNet is
   exactly the XLinear self branch.

2) RNG-safe dependency initialization:
   --hsd_rng_safe_init 1 restores RNG states after creating the dependency
   branch, so the self branch training trajectory remains aligned with XLinear
   as much as possible.

3) Residual-targeted auxiliary loss:
   get_aux_loss(target=batch_y) adds
       hsd_residual_target_lambda * MSE(delta_raw, stopgrad(y_true - y_self))
   plus a small correction magnitude regularizer.

4) Bounded final correction:
   the learned raw delta is clipped through tanh(delta/scale)*scale before being
   multiplied by sample risk and horizon gate.

Input shape : x_enc [B, L, C]
Output shape: [B, H, C] for M, [B, H, 1] for MS/S
"""

import math
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.XLinear import Forcast_multi, Forcast_with_exogenous


class StateEncoder(nn.Module):
    """Encode the current window state from robust shape statistics."""

    def __init__(self, seq_len: int, channels: int, state_dim: int, dropout: float = 0.0):
        super().__init__()
        self.seq_len = seq_len
        self.channels = channels
        in_dim = 6 * channels  # mean, std, trend, last, half_shift, range
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
        L = x.shape[1]
        mid = max(1, L // 2)
        mean = x.mean(dim=1)
        std = torch.sqrt(torch.var(x, dim=1, unbiased=False) + 1e-5)
        trend = x[:, -1, :] - x[:, 0, :]
        last = x[:, -1, :]
        half_shift = x[:, mid:, :].mean(dim=1) - x[:, :mid, :].mean(dim=1)
        range_stat = x.max(dim=1).values - x.min(dim=1).values
        stats = torch.cat([mean, std, trend, last, half_shift, range_stat], dim=-1)
        return self.net(stats)


class ShapeRiskGate(nn.Module):
    """Non-parametric sample-level risk score for dependency correction."""

    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        self.use_quantile = int(getattr(configs, 'hsd_risk_use_quantile', 1))
        self.quantile = float(getattr(configs, 'hsd_risk_quantile', 0.70))
        self.threshold = float(getattr(configs, 'hsd_risk_threshold', 1.00))
        self.sharpness = float(getattr(configs, 'hsd_risk_sharpness', 8.0))
        self.floor = float(getattr(configs, 'hsd_risk_floor', 0.0))
        self.detach_risk = int(getattr(configs, 'hsd_detach_risk', 1))

    def forward(self, x: torch.Tensor, out_channels: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return horizon-level risk and scalar risk.

        Args:
            x: [B, L, C], normalized input.
        Returns:
            risk_h: [B, H, out_channels]
            risk_scalar: [B, 1]
        """
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


class HorizonStateDependencyBranch(nn.Module):
    """State-conditioned, horizon-aware dependency residual branch."""

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
        self.gate_init = float(getattr(configs, 'hsd_gate_init', -4.0))
        self.temperature = max(float(getattr(configs, 'hsd_temperature', 1.0)), 1e-6)
        self.zero_init = int(getattr(configs, 'hsd_zero_init', 1))
        self.residual_scale = max(float(getattr(configs, 'hsd_residual_scale', 0.08)), 1e-6)
        self.aux_lambda = float(getattr(configs, 'hsd_aux_lambda', 0.0))

        self.state_encoder = StateEncoder(
            seq_len=self.seq_len,
            channels=self.channels,
            state_dim=self.state_dim,
            dropout=self.dropout_p,
        )
        self.risk_gate = ShapeRiskGate(configs)

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

        # This predicts raw residual in normalized output space.  Zero init keeps
        # initial prediction identical to XLinear, but residual-target loss gives
        # a direct nonzero gradient to this layer.
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
        self.last_aux_reg: Optional[torch.Tensor] = None
        self.last_debug: Dict[str, float] = {}
        self.last_raw_delta: Optional[torch.Tensor] = None
        self.last_correction: Optional[torch.Tensor] = None
        self.last_risk: Optional[torch.Tensor] = None

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
        if self.topk <= 0 or self.topk >= scores.shape[-1]:
            return scores
        vals, idx = torch.topk(scores, k=self.topk, dim=-1)
        masked = torch.full_like(scores, float('-inf'))
        masked.scatter_(-1, idx, vals)
        return masked

    def forward(self, x: torch.Tensor):
        """Return correction, raw_delta, gate and risk in normalized space.

        x: [B, L, C]
        """
        b, _, c = x.shape
        assert c == self.channels, f"Expected {self.channels} channels, got {c}"

        x_ch = x.permute(0, 2, 1)       # [B, C, L]
        z = self.value_proj(x_ch)       # [B, C, d]
        state = self.state_encoder(x)   # [B, state_dim]

        keys = self.key_norm(self.key_proj(z))
        base_queries = self.query_proj(z)
        target_queries = base_queries if self.feature == 'M' else base_queries[:, -1:, :]

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

        # Raw residual per source variable and horizon: [B, C, H]
        src_raw = self.source_linear(self.source_dropout(z))

        raw_chunks = []
        gate_chunks = []
        group_gate = torch.sigmoid(self.gate_head(state))  # [B, G]
        for g, (start, end) in enumerate(self._groups):
            src_g = src_raw[:, :, start:end]                      # [B, C, Hg]
            attn_g = attn[:, g, :, :]                             # [B, T, C]
            raw_g = torch.einsum('btc,bch->bth', attn_g, src_g)   # [B, T, Hg]
            raw_chunks.append(raw_g)

            gate_g = group_gate[:, g].view(b, 1, 1).expand(b, self.out_channels, end - start)
            gate_chunks.append(gate_g)

        raw_delta = torch.cat(raw_chunks, dim=-1).permute(0, 2, 1).contiguous()  # [B,H,T]
        gate = torch.cat(gate_chunks, dim=-1).permute(0, 2, 1).contiguous()      # [B,H,T]
        risk, risk_scalar = self.risk_gate(x, out_channels=self.out_channels)

        # Near zero, bounded_delta ≈ raw_delta; outside range, it is clipped.
        bounded_delta = torch.tanh(raw_delta / self.residual_scale) * self.residual_scale
        correction = risk * gate * bounded_delta

        self.last_raw_delta = raw_delta
        self.last_correction = correction
        self.last_risk = risk
        self.last_aux_reg = self.aux_lambda * correction.pow(2).mean() if (self.training and self.aux_lambda > 0) else None

        with torch.no_grad():
            self.last_debug = {
                'risk_mean': float(risk_scalar.mean().detach().cpu()),
                'gate_mean': float(gate.mean().detach().cpu()),
                'raw_delta_abs_mean': float(raw_delta.abs().mean().detach().cpu()),
                'correction_abs_mean': float(correction.abs().mean().detach().cpu()),
            }
        return correction, raw_delta, gate, risk


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
        self.disable_dep = int(getattr(configs, 'hsd_disable_dep', 0))
        self.rng_safe_init = int(getattr(configs, 'hsd_rng_safe_init', 1))
        self.hsd_debug = int(getattr(configs, 'hsd_debug', 0))
        self._debug_counter = 0

        self.residual_target_lambda = float(getattr(configs, 'hsd_residual_target_lambda', 0.05))
        self.residual_target_clip = float(getattr(configs, 'hsd_residual_target_clip', 3.0))
        self.residual_target_risk_weight = int(getattr(configs, 'hsd_residual_target_risk_weight', 1))
        self.residual_target_min_weight = float(getattr(configs, 'hsd_residual_target_min_weight', 0.10))

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
            print('[HSDNet-v3] dependency branch is NOT instantiated. '
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

        # Cached tensors from the latest forward pass, used by residual-target loss.
        self._last_means = None
        self._last_stdev = None
        self._last_y_self_norm = None
        self._last_raw_delta_norm = None
        self._last_correction_norm = None
        self._last_risk = None

    def _self_forecast(self, x_norm: torch.Tensor) -> torch.Tensor:
        x_ch = x_norm.permute(0, 2, 1)
        en = self.self_backbone(x_ch)
        y_self = self.self_head(en).permute(0, 2, 1).contiguous()
        return y_self

    def _forecast_normed(self, x_norm: torch.Tensor) -> torch.Tensor:
        y_self = self._self_forecast(x_norm)
        self._last_y_self_norm = y_self

        if self.disable_dep:
            self._last_raw_delta_norm = None
            self._last_correction_norm = None
            self._last_risk = None
            return y_self

        correction, raw_delta, _, risk = self.dep_branch(x_norm)
        self._last_raw_delta_norm = raw_delta
        self._last_correction_norm = correction
        self._last_risk = risk
        y = y_self + correction

        if self.hsd_debug and self.training:
            # Print once at the beginning and then every 100 forwards.  The first
            # line may still be zero because of zero-init; later lines show whether
            # the residual branch is actually learning.
            if self._debug_counter == 0 or self._debug_counter % 100 == 0:
                dbg = getattr(self.dep_branch, 'last_debug', {})
                print('[HSDNet-v3 DEBUG] ' + ', '.join([f'{k}={v:.6f}' for k, v in dbg.items()]))
            self._debug_counter += 1
        return y

    def get_aux_loss(self, target: Optional[torch.Tensor] = None):
        """Return residual-target loss plus correction regularization.

        Args:
            target: ground-truth output in original scale and already sliced in
                    exp_main.py to match model output, shape [B,H,Cout].
        """
        if self.disable_dep or self.dep_branch is None:
            return None

        losses = []
        reg_loss = getattr(self.dep_branch, 'last_aux_reg', None)
        if reg_loss is not None:
            losses.append(reg_loss)

        if (target is not None and self.training and self.residual_target_lambda > 0
                and self._last_raw_delta_norm is not None and self._last_y_self_norm is not None):
            # Convert target from original scale to the normalized space used by
            # y_self/raw_delta.
            if self.norm:
                if self.feature == 'M':
                    mean = self._last_means[:, 0, :].unsqueeze(1)
                    std = self._last_stdev[:, 0, :].unsqueeze(1)
                else:
                    mean = self._last_means[:, 0, -1:].unsqueeze(1)
                    std = self._last_stdev[:, 0, -1:].unsqueeze(1)
                target_norm = (target - mean) / (std + 1e-5)
            else:
                target_norm = target

            residual_target = (target_norm - self._last_y_self_norm.detach()).detach()
            if self.residual_target_clip > 0:
                residual_target = torch.clamp(
                    residual_target,
                    min=-self.residual_target_clip,
                    max=self.residual_target_clip,
                )

            err = (self._last_raw_delta_norm - residual_target).pow(2)
            if self.residual_target_risk_weight and self._last_risk is not None:
                weight = self.residual_target_min_weight + self._last_risk.detach()
                err = err * weight
            residual_loss = self.residual_target_lambda * err.mean()
            losses.append(residual_loss)

        if not losses:
            return None
        return sum(losses)

    def get_debug_stats(self):
        if self.disable_dep or self.dep_branch is None:
            return {}
        dbg = dict(getattr(self.dep_branch, 'last_debug', {}))
        if self._last_raw_delta_norm is not None:
            with torch.no_grad():
                dbg['raw_delta_abs_mean_cached'] = float(self._last_raw_delta_norm.abs().mean().detach().cpu())
        return dbg

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
