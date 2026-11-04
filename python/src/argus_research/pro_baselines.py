"""Additional recurrent control for the exploratory Pro comparison."""

from torch import Tensor, nn

from .ahem_v6 import FlowLaneLstmNextPacket


class PersistenceResidualLstm(FlowLaneLstmNextPacket):
    """Same LSTM/parameter count, initialized at current-packet persistence.

    The head learns additive corrections in the existing log-feature space.
    Outputs are clamped to nonnegative values. This is a separate comparator,
    not a modification to previously evaluated LSTM checkpoints.
    """

    def __init__(self, packet_dim: int = 12, hidden_dim: int = 255):
        super().__init__(packet_dim=packet_dim, hidden_dim=hidden_dim)
        nn.init.zeros_(self.predict[-1].weight)
        nn.init.zeros_(self.predict[-1].bias)

    def forward(self, packets: Tensor) -> Tensor:
        if (
            packets.ndim != 3
            or packets.shape[1] == 0
            or packets.shape[-1] != self.packet_dim + 1
        ):
            raise ValueError("residual LSTM expects nonempty packet+relation windows")
        if not (packets[:, -1, -1] == 1).all():
            raise ValueError("residual LSTM query must be the final event")
        return (packets[:, -1, :2] + super().forward(packets)).clamp_min(0)

    def forward_step(self, packet: Tensor, state: tuple[Tensor, Tensor] | None = None):
        correction, next_state = super().forward_step(packet, state)
        current = packet if packet.ndim == 2 else packet[:, -1]
        return (current[:, :2] + correction).clamp_min(0), next_state
