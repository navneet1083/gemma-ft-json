"""gemma_ft_json — offline image-to-JSON fine-tuning of Gemma 3 270M.

Plug-and-play layout (each sub-package is independently importable):

    gemma_ft_json.config     -> YAML config loading & validation
    gemma_ft_json.utils      -> device/seed/logging/registry helpers
    gemma_ft_json.data       -> synthetic table generator, Dataset, collator
    gemma_ft_json.models     -> vision encoder, projector, LoRA, fused VLM
    gemma_ft_json.training   -> trainer, losses, checkpointing
    gemma_ft_json.inference  -> predictor with JSON guard
"""

__version__ = "0.1.0"

from .exceptions import (  # noqa: F401
    GemmaFTError, ConfigError, ModelLoadError,
    ShapeError, NumericalError, DataError, CheckpointError,
)
