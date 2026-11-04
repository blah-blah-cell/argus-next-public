"""Dense execution of the existing unidirectional residual-LSTM control.

Right padding cannot affect earlier outputs of a unidirectional recurrence.
Selecting the last real output is equivalent to the packed final hidden state.
The checkpoint tensor shapes and forecast head are unchanged. This is a CPU
training optimization, not a new architecture or a changed forecasting task.
"""

import torch
from torch import Tensor, nn

from .pro_baselines import PersistenceResidualLstm


class DenseResidualLstm(PersistenceResidualLstm):
    def forward(self, packets: Tensor) -> Tensor:
        if (
            packets.ndim != 3
            or packets.shape[0] == 0
            or packets.shape[1] == 0
            or packets.shape[-1] != self.packet_dim + 1
        ):
            raise ValueError("dense LSTM expects nonempty packet+relation windows")
        if not (packets[:, -1, -1] == 1).all():
            raise ValueError("dense LSTM query must be the final event")
        relation = packets[..., -1] > 0.5
        lengths = relation.sum(-1)
        lanes = [
            row[mask, : self.packet_dim]
            for row, mask in zip(packets, relation, strict=True)
        ]
        padded = nn.utils.rnn.pad_sequence(lanes, batch_first=True)
        sequence, _ = self.encoder(padded)
        hidden = sequence[
            torch.arange(len(packets), device=packets.device), lengths - 1
        ]
        return (packets[:, -1, :2] + self.predict(hidden)).clamp_min(0)
