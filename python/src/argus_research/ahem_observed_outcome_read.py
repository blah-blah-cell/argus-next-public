"""Research-only causal outcome read. Not an admitted application model."""
import torch


def observed_outcome_read(weights, successors, valid, current):
    """Read completed historical outcomes, or persist the observed query packet.

    This function has no future target argument, learned timing replacement,
    residual output transform, or target-dependent choice. The caller must
    supply the causal eligibility mask produced by its bounded memory reader.
    """
    if (weights.ndim != 2 or successors.shape != (*weights.shape, 2)
            or valid.shape != weights.shape or valid.dtype != torch.bool
            or current.shape != (len(weights), 2)):
        raise ValueError('unaligned outcome read')
    if not weights.is_floating_point() or not len(weights):
        raise ValueError('nonempty floating point read required')
    for tensor in (successors, current):
        if tensor.device != weights.device or tensor.dtype != weights.dtype:
            raise ValueError('outcome tensors must share dtype and device')
    if valid.device != weights.device:
        raise ValueError('eligibility mask must share device')
    if any(not torch.isfinite(tensor).all() for tensor in (weights, successors, current)):
        raise ValueError('nonfinite outcome read')
    if (weights < 0).any() or (weights[~valid] != 0).any():
        raise ValueError('negative or ineligible retrieval weight')
    available = valid.any(-1)
    expected = available.to(weights.dtype)
    if not torch.allclose(weights.sum(-1), expected, rtol=1e-5, atol=1e-6):
        raise ValueError('retrieval mass must be one, or zero without memory')
    mixture = (weights[..., None] * successors).sum(1)
    return torch.where(available[:, None], mixture, current)
