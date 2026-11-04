"""Compact diagonal state-space baseline for long packet sequences.

This is an S4D-style diagonal continuous-time state space discretized into a
causal convolution and evaluated with an FFT.  It is a real long-range
baseline, not a Mamba substitute or an Argus memory component.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class S4DLayer(nn.Module):
    """Diagonal SSM convolution following the S4D parameterization."""

    def __init__(
        self,
        channels: int,
        state_dim: int = 32,
        *,
        minimum_step: float = 1e-3,
        maximum_step: float = 1e-1,
    ) -> None:
        super().__init__()
        if channels <= 0 or state_dim <= 0:
            raise ValueError("S4D dimensions must be positive")
        if not 0.0 < minimum_step < maximum_step:
            raise ValueError("S4D step bounds are invalid")
        self.channels = channels
        self.state_dim = state_dim
        self.log_step = nn.Parameter(
            torch.empty(channels).uniform_(
                math.log(minimum_step), math.log(maximum_step)
            )
        )
        self.log_a_real = nn.Parameter(torch.full((channels, state_dim), -0.5))
        frequencies = math.pi * torch.arange(state_dim, dtype=torch.float32)
        self.a_imag = nn.Parameter(frequencies[None].repeat(channels, 1))
        scale = 1.0 / math.sqrt(state_dim)
        self.c_real = nn.Parameter(torch.randn(channels, state_dim) * scale)
        self.c_imag = nn.Parameter(torch.randn(channels, state_dim) * scale)
        self.skip = nn.Parameter(torch.ones(channels))

    def kernel(
        self, length: int, *, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        if length <= 0:
            raise ValueError("S4D kernel length must be positive")
        # Kernel construction remains float32/complex64 under mixed precision;
        # the resulting real kernel is cast back to the input dtype.
        step = self.log_step.float().exp()
        a = torch.complex(-self.log_a_real.float().exp(), self.a_imag.float())
        c = torch.complex(self.c_real.float(), self.c_imag.float())
        delta_a = step[:, None] * a
        c_bar = c * torch.expm1(delta_a) / a
        positions = torch.arange(length, device=device, dtype=torch.float32)
        powers = torch.exp(delta_a[:, :, None] * positions[None, None, :])
        values = 2.0 * torch.einsum("hn,hnl->hl", c_bar, powers).real
        return values.to(dtype=dtype)

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 3 or values.size(-1) != self.channels:
            raise ValueError("S4D input has an incompatible shape")
        length = values.size(1)
        output_dtype = values.dtype
        fft_dtype = (
            torch.float32
            if values.dtype in (torch.float16, torch.bfloat16)
            else values.dtype
        )
        channels_first = values.transpose(1, 2).to(dtype=fft_dtype)
        kernel = self.kernel(length, device=values.device, dtype=fft_dtype)
        transform_length = 2 * length
        signal_frequency = torch.fft.rfft(channels_first, n=transform_length)
        kernel_frequency = torch.fft.rfft(kernel, n=transform_length)
        convolved = torch.fft.irfft(
            signal_frequency * kernel_frequency[None], n=transform_length
        )[..., :length]
        convolved = convolved + channels_first * self.skip[None, :, None]
        return convolved.transpose(1, 2).to(dtype=output_dtype)


class S4DBlock(nn.Module):
    def __init__(self, channels: int, state_dim: int) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(channels)
        self.ssm = S4DLayer(channels, state_dim)
        self.mix = nn.Linear(channels, channels)

    def forward(self, values: Tensor) -> Tensor:
        residual = values
        values = self.ssm(self.normalization(values))
        return residual + self.mix(F.gelu(values))


class S4DNextPacket(nn.Module):
    """Parameter-scale S4D baseline for causal next-packet forecasting."""

    def __init__(
        self,
        feature_dim: int,
        *,
        output_dim: int | None = None,
        channels: int = 64,
        state_dim: int = 32,
        layers: int = 3,
    ) -> None:
        super().__init__()
        output_dim = feature_dim if output_dim is None else output_dim
        if feature_dim <= 0 or output_dim <= 0 or layers <= 0:
            raise ValueError("S4D model dimensions must be positive")
        self.input = nn.Linear(feature_dim, channels)
        self.blocks = nn.ModuleList(
            S4DBlock(channels, state_dim) for _ in range(layers)
        )
        self.normalization = nn.LayerNorm(channels)
        self.predict = nn.Sequential(
            nn.Linear(channels, 64),
            nn.GELU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, context: Tensor) -> Tensor:
        values = self.input(context)
        for block in self.blocks:
            values = block(values)
        return self.predict(self.normalization(values[:, -1]))
