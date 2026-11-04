"""Small causal baselines and the fixed-chunk AHEM-F reference model."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class GruClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 96) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, packets: Tensor) -> Tensor:
        state, _ = self.gru(packets)
        return self.head(state[:, -1])


class CausalTcnClassifier(nn.Module):
    def __init__(self, input_dim: int, channels: int = 96) -> None:
        super().__init__()
        self.input = nn.Conv1d(input_dim, channels, 1)
        self.blocks = nn.ModuleList(
            [
                nn.Conv1d(channels, channels, 3, dilation=d, padding=d)
                for d in (1, 2, 4, 8)
            ]
        )
        self.head = nn.Linear(channels, 2)

    def forward(self, packets: Tensor) -> Tensor:
        state = self.input(packets.transpose(1, 2))
        for conv in self.blocks:
            residual = state
            state = F.gelu(conv(state)[:, :, : state.size(2)])
            state = state + residual
        return self.head(state[:, :, -1])


class AhemFixedClassifier(nn.Module):
    """Bounded causal chunk retrieval; no full-history attention."""

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 64,
        key_dim: int = 32,
        chunk_size: int = 64,
        top_k: int = 8,
    ) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.top_k = top_k
        self.packet = nn.Sequential(nn.Linear(input_dim, embedding_dim), nn.GELU())
        self.query = nn.Linear(embedding_dim, key_dim)
        self.key = nn.Linear(embedding_dim, key_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        # A causal writer decides which already-observed packet evidence a
        # fixed-size chunk retains.  This is not query-conditioned attention:
        # the chunk is written before any later query exists.
        self.writer_salience = nn.Linear(embedding_dim, 1)
        self.head = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, 2),
        )

    def forward(self, packets: Tensor) -> Tensor:
        encoded = self.packet(packets)
        query = self.query(encoded[:, -1])
        # Current query packet may not enter a completed-memory chunk.
        history = encoded[:, :-1]
        chunks = history.size(1) // self.chunk_size
        if chunks == 0:
            raise ValueError("sequence too short for one completed chunk")
        history = history[:, : chunks * self.chunk_size]
        chunk_packets = history.reshape(history.size(0), chunks, self.chunk_size, -1)
        writer_weights = self.writer_salience(chunk_packets).squeeze(-1).softmax(dim=2)
        summaries = (writer_weights.unsqueeze(-1) * chunk_packets).sum(dim=2)
        keys = self.key(summaries)
        values = self.value(summaries)
        scores = torch.einsum("bd,bmd->bm", query, keys) / math.sqrt(keys.size(-1))
        selected_scores, selected = scores.topk(min(self.top_k, chunks), dim=1)
        selected_values = values.gather(
            1, selected.unsqueeze(-1).expand(-1, -1, values.size(-1))
        )
        weights = selected_scores.softmax(dim=1).unsqueeze(-1)
        retrieved = (weights * selected_values).sum(dim=1)
        return self.head(torch.cat((encoded[:, -1], retrieved), dim=-1))


class CausalRouter(nn.Module):
    """Chooses among frozen experts using only cheap pre-expert observations."""

    def __init__(self, input_dim: int, expert_count: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32), nn.GELU(), nn.Linear(32, expert_count)
        )

    def forward(self, causal_summary: Tensor) -> Tensor:
        return self.network(causal_summary)


def causal_router_features(packets: Tensor) -> Tensor:
    """Last packet + causal prefix mean.  No expert scores or future packets."""
    return torch.cat((packets[:, -1], packets[:, :-1].mean(dim=1)), dim=1)
