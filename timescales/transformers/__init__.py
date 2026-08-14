"""Small transformer and linear-sequence models for timescale experiments."""

from .sequence_models import (
    AttentionSequenceModel,
    LinearFIR,
    StaticLinearAttention,
    create_sequence_model,
)
from .lightning import TransformerSequenceLightning

__all__ = [
    "AttentionSequenceModel",
    "LinearFIR",
    "StaticLinearAttention",
    "TransformerSequenceLightning",
    "create_sequence_model",
]
