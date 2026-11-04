"""Parameter-matched competitors for the AHEM v6.1 secondary comparison.

These models are trained from scratch on the same development sources with the
same forecast-only training contract, then evaluated on the frozen Friday-23
coordinates.  They are secondary evidence only: the holdout is no longer
untouched after the one-attempt final evaluation.

Threat ranking (from NOVELTY_MATRIX.md and LITERATURE_PROTOCOL.md):

1. Mamba-style selective SSM — an input-selective state-space challenger. It
   is not the official Mamba implementation and cannot satisfy an official
   Mamba-baseline requirement by itself.
2. S4D — diagonal state space; frozen in the v2 matrix but never executed.
3. Long TCN — dilated causal CNN with >30k-packet receptive field.
4. GRU — single-gate recurrent companion to the beaten LSTM.
5. DLinear — a deliberately simple linear-plus-trend baseline.
"""

from __future__ import annotations


import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .ahem_pretrain import GruNextPacket, LongTcnNextPacket
from .s4d_baseline import S4DNextPacket

FEATURE_DIM = 13  # 12 causal packet features + 1 relation bit
OUTPUT_DIM = 2  # log-frame-length and log-per-flow-interarrival

# Parameter-matched configurations (target ~41k, the frozen candidate budget).
MATCHED_CONFIGS = {
    "gru": {"hidden_dim": 96},
    "s4d": {"channels": 64, "state_dim": 32, "layers": 3},
    "long_tcn": {"channels": 28},
}


class SelectiveSsmLayer(nn.Module):
    """A minimal pure-PyTorch Mamba-style selective state-space scan.

    Unlike S4D's fixed transition, B/C/delta depend on the input — the core
    selectivity that distinguishes Mamba from diagonal SSMs.  This is not the
    official Mamba implementation (which needs a specialized scan kernel). It
    is a minimal Mamba-style ablation, not a drop-in reproduction of Mamba.
    Training on a CUDA GPU is recommended because the Python loop over L
    positions is serial.
    """

    def __init__(self, channels: int, state_dim: int, *, expand: int = 2) -> None:
        super().__init__()
        inner = channels * expand
        self.norm = nn.LayerNorm(channels)
        self.proj_in = nn.Linear(channels, inner * 2, bias=False)  # gate + value
        self.conv = nn.Conv1d(inner, inner, 4, padding=3, groups=inner)
        # Input-dependent projections: B, C, and delta from the projected input.
        self.b_proj = nn.Linear(inner, state_dim, bias=False)
        self.c_proj = nn.Linear(inner, state_dim, bias=False)
        self.delta_proj = nn.Linear(inner, 1, bias=True)
        # A is a learned log-negative parameter per (inner, state) pair.
        self.log_a = nn.Parameter(torch.randn(inner, state_dim) * 0.01 - 4.0)
        self.norm_out = nn.LayerNorm(inner)
        self.proj_out = nn.Linear(inner, channels, bias=False)
        self.state_dim = state_dim
        self.channels = channels
        self.expand = expand
        self.inner = inner

    def forward(self, x: Tensor) -> Tensor:
        batch, length, _ = x.shape
        h = self.norm(x)
        projected = self.proj_in(h)  # [B, L, 2*inner]
        gate, value = projected.chunk(2, dim=-1)  # each [B, L, inner]
        value = self.conv(value.transpose(1, 2)).transpose(1, 2)[:, :-1]
        value = F.pad(value, (0, 0, 1, 0))  # causal shift
        value = F.silu(value)
        # Input-dependent SSM parameters at every position.
        b = self.b_proj(value)  # [B, L, S]
        c = self.c_proj(value)  # [B, L, S]
        delta = F.softplus(self.delta_proj(value).squeeze(-1))  # [B, L]
        a = -torch.exp(self.log_a)  # [inner, S], negative for stable decay
        # Sequential scan: state is [B, inner, S], input is [B, inner] at each t.
        state = torch.zeros(
            batch, self.inner, self.state_dim, device=x.device, dtype=x.dtype
        )
        outputs: list[Tensor] = []
        for t in range(length):
            a_bar = torch.exp(delta[:, t : t + 1, None] * a[None])  # [B, inner, S]
            state = a_bar * state + b[:, t, None, :] * value[:, t, :, None]
            y_t = (state * c[:, t, None, :]).sum(dim=-1)  # [B, inner]
            outputs.append(y_t)
        y = torch.stack(outputs, dim=1)  # [B, L, inner]
        y = y * F.silu(gate)
        y = self.norm_out(y)
        return x + self.proj_out(y)


class MambaNextPacket(nn.Module):
    """Parameter-matched Mamba-style selective SSM for next-packet forecasting."""

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        *,
        output_dim: int = OUTPUT_DIM,
        channels: int = 48,
        state_dim: int = 12,
        layers: int = 2,
    ) -> None:
        super().__init__()
        self.input = nn.Linear(feature_dim, channels)
        self.blocks = nn.ModuleList(
            SelectiveSsmLayer(channels, state_dim) for _ in range(layers)
        )
        self.norm = nn.LayerNorm(channels)
        self.predict = nn.Sequential(
            nn.Linear(channels, 64),
            nn.GELU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, context: Tensor) -> Tensor:
        x = self.input(context)
        for block in self.blocks:
            x = block(x)
        return self.predict(self.norm(x[:, -1]))


class DLinearNextPacket(nn.Module):
    """A deliberately simple linear-plus-trend baseline.

    Decomposes the local 128-packet window into a trend (moving average) and
    seasonal (residual) component, applies independent linear projections to
    each, and sums.  No nonlinearity.  If this matches AHEM, the memory
    mechanism is unnecessary for the forecasting task.
    """

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        *,
        output_dim: int = OUTPUT_DIM,
        kernel_size: int = 25,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.kernel_size = kernel_size
        # Trend: linear map from averaged features to output.
        self.trend_proj = nn.Linear(feature_dim, output_dim)
        # Seasonal: linear map from residual features to output.
        self.seasonal_proj = nn.Linear(feature_dim, output_dim)

    def _moving_average(self, x: Tensor) -> Tensor:
        # x: [B, L, D] -> averaged: [B, L, D] with causal moving average
        dim = x.size(-1)
        pad = self.kernel_size - 1
        weight = (
            torch.ones(dim, 1, self.kernel_size, device=x.device) / self.kernel_size
        )
        averaged = F.conv1d(x.transpose(1, 2), weight, padding=pad, groups=dim)
        return averaged[:, :, : x.size(1)].transpose(1, 2)

    def forward(self, context: Tensor) -> Tensor:
        trend = self._moving_average(context)
        seasonal = context - trend
        trend_out = self.trend_proj(trend[:, -1])
        seasonal_out = self.seasonal_proj(seasonal[:, -1])
        return trend_out + seasonal_out


def construct_competitor(name: str) -> nn.Module:
    """Build a parameter-matched competitor by name."""
    if name == "gru":
        return GruNextPacket(FEATURE_DIM, hidden_dim=96, output_dim=OUTPUT_DIM)
    if name == "s4d":
        return S4DNextPacket(
            FEATURE_DIM,
            channels=64,
            state_dim=32,
            layers=3,
            output_dim=OUTPUT_DIM,
        )
    if name == "long_tcn":
        return LongTcnNextPacket(FEATURE_DIM, channels=28, output_dim=OUTPUT_DIM)
    if name == "mamba":
        return MambaNextPacket(FEATURE_DIM, channels=48, state_dim=12, layers=2)
    if name == "dlinear":
        return DLinearNextPacket(FEATURE_DIM, output_dim=OUTPUT_DIM)
    raise ValueError(f"unknown competitor: {name}")


def competitor_parameters(name: str) -> int:
    return sum(p.numel() for p in construct_competitor(name).parameters())
