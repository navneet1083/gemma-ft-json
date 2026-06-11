from .guards import assert_shape, check_finite  # noqa: F401
from .vision_encoder import TableViTLite, VISION_ENCODERS  # noqa: F401
from .projector import RMSCalibratedProjector, PROJECTORS  # noqa: F401
from .lora import LoRALinear, inject_lora, lora_state_dict, mark_only_lora_trainable  # noqa: F401
# gemma_loader / vlm require `transformers`. Import lazily so the vision
# encoder, projector and LoRA stay importable in lightweight environments
# (plug-and-play: these submodules have no LLM dependency by design).
try:  # pragma: no cover - environment dependent
    from .gemma_loader import load_gemma_local  # noqa: F401
    from .vlm import GemmaVisionForJSON  # noqa: F401
except ModuleNotFoundError as _e:  # transformers not installed
    if _e.name != "transformers":
        raise
