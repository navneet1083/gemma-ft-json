"""Custom exception hierarchy.

Every layer of the codebase raises *typed* exceptions instead of bare
``Exception`` so that callers (notebooks, scripts, future services) can
catch precisely what they can handle and let the rest propagate.
"""


class GemmaFTError(Exception):
    """Base class for every error raised by this package."""


class ConfigError(GemmaFTError):
    """Raised when the YAML config is missing/invalid or a path is wrong."""


class ModelLoadError(GemmaFTError):
    """Raised when local Gemma weights/tokenizer cannot be loaded offline."""


class ShapeError(GemmaFTError):
    """Raised by dimension guards when a tensor has an unexpected shape."""


class NumericalError(GemmaFTError):
    """Raised when NaN/Inf are detected in activations, loss or gradients."""


class DataError(GemmaFTError):
    """Raised for missing/corrupt dataset files or malformed manifests."""


class CheckpointError(GemmaFTError):
    """Raised when checkpoint save/load/resume fails."""
