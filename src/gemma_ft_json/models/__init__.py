"""Modeling subpackage: vision encoder, projector, decoder backends, VLM, losses."""
from .vision_encoder import build_vision_encoder, FromScratchViT, LocalSiglipEncoder
from .projector import build_projector, MLPProjector
from .backends import build_decoder, TinyStubDecoder, LocalGemmaDecoder, inject_lora, LoRALinear
from .losses import faithful_lm_loss
from .vlm import GemmaTableVLM, build_model

__all__ = [
    "build_vision_encoder", "FromScratchViT", "LocalSiglipEncoder",
    "build_projector", "MLPProjector",
    "build_decoder", "TinyStubDecoder", "LocalGemmaDecoder", "inject_lora", "LoRALinear",
    "faithful_lm_loss",
    "GemmaTableVLM", "build_model",
]
