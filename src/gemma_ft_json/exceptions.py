"""Custom exception hierarchy.

A small, specific exception tree lets callers catch exactly what they expect and
makes logs self-explanatory. Every module raises one of these rather than a bare
`Exception`.
"""
from __future__ import annotations


class GemmaFTError(Exception):
    """Base class for all errors raised by this package."""


class ConfigError(GemmaFTError):
    """Invalid/missing configuration or config file."""


class DataError(GemmaFTError):
    """Problem building, reading, or validating the dataset/manifest."""


class WeightsNotFoundError(GemmaFTError):
    """A required LOCAL weights directory/file is missing.

    Raised INSTEAD of silently downloading from Hugging Face (forbidden).
    """


class ShapeMismatchError(GemmaFTError):
    """A tensor had an unexpected shape; raised by the dimension guards so a bad
    wiring fails loudly *before* it produces NaNs deep inside training."""


class NumericalError(GemmaFTError):
    """NaN/Inf detected in a tensor (activations, loss, or gradients)."""


class CheckpointError(GemmaFTError):
    """Failure while saving/loading a training checkpoint."""
