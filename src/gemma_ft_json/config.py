"""Configuration loading and validation.

We parse YAML into typed dataclasses for attribute access, central validation,
and easy re-serialization (a "snapshot" saved next to checkpoints so inference
can rebuild the exact same model). Unknown keys raise early — typos fail loudly.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List

import yaml

from .exceptions import ConfigError


@dataclass
class ProjectCfg:
    name: str = "gemma-ft-json"
    seed: int = 42
    device: str = "auto"


@dataclass
class PathsCfg:
    images_dir: str = ""
    json_dir: str = ""
    pdf_dir: str = ""
    manifest_dir: str = "./data/manifests"
    runs_dir: str = "./runs"


@dataclass
class DecoderCfg:
    backend: str = "stub"
    local_dir: str = ""
    hidden_size: int = 640
    vocab_size: int = 262144
    max_seq_len: int = 2048
    n_structure_tokens: int = 8
    n_loc_bins: int = 0


@dataclass
class VisionCfg:
    backend: str = "scratch_vit"
    local_dir: str = ""
    image_size: int = 384
    patch_size: int = 16
    embed_dim: int = 384
    depth: int = 6
    num_heads: int = 6
    freeze: bool = True


@dataclass
class ProjectorCfg:
    type: str = "mlp"
    inner_mult: int = 4
    dropout: float = 0.0


@dataclass
class LoraCfg:
    enabled: bool = True
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])


@dataclass
class ModelCfg:
    decoder: DecoderCfg = field(default_factory=DecoderCfg)
    vision: VisionCfg = field(default_factory=VisionCfg)
    projector: ProjectorCfg = field(default_factory=ProjectorCfg)
    lora: LoraCfg = field(default_factory=LoraCfg)


@dataclass
class DataCfg:
    prompt: str = "Extract the table as JSON."
    max_target_tokens: int = 1024
    val_fraction: float = 0.1
    require_valid_json: bool = True
    use_pymupdf_boxes: bool = False


@dataclass
class DataLoaderCfg:
    batch_size: int = 2
    num_workers: int = 2
    pin_memory: bool = False
    shuffle_train: bool = True


@dataclass
class LossCfg:
    label_smoothing: float = 0.0
    content_token_weight: float = 1.5
    grounding_weight: float = 0.0


@dataclass
class OptimCfg:
    name: str = "adamw"
    lr: float = 2.0e-4
    weight_decay: float = 0.01
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1.0e-8
    grad_clip_norm: float = 1.0
    scheduler: str = "cosine"
    warmup_steps: int = 100


@dataclass
class TrainingCfg:
    epochs: int = 5
    max_steps: int = -1
    grad_accum_steps: int = 4
    log_every_steps: int = 10
    eval_every_steps: int = 200
    save_best_only: bool = True
    resume: bool = True
    amp: bool = False


@dataclass
class LoggingCfg:
    level: str = "INFO"
    metrics_filename: str = "metrics.jsonl"
    log_filename: str = "train.log"


@dataclass
class Config:
    project: ProjectCfg = field(default_factory=ProjectCfg)
    paths: PathsCfg = field(default_factory=PathsCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    data: DataCfg = field(default_factory=DataCfg)
    dataloader: DataLoaderCfg = field(default_factory=DataLoaderCfg)
    loss: LossCfg = field(default_factory=LossCfg)
    optim: OptimCfg = field(default_factory=OptimCfg)
    training: TrainingCfg = field(default_factory=TrainingCfg)
    logging: LoggingCfg = field(default_factory=LoggingCfg)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_snapshot(self, path: str | Path) -> None:
        """Write the *effective* config to YAML for reproducibility."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)


def _merge_into(dc_instance, data: Dict[str, Any]):
    """Recursively overlay `data` (from YAML) onto a dataclass instance."""
    if data is None:
        return dc_instance
    for key, value in data.items():
        if not hasattr(dc_instance, key):
            raise ConfigError(f"Unknown config key '{key}' for {type(dc_instance).__name__}")
        current = getattr(dc_instance, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_into(current, value)
        else:
            setattr(dc_instance, key, value)
    return dc_instance


def _apply_dotted_overrides(cfg: "Config", overrides: List[str]) -> None:
    """Apply CLI overrides like 'training.epochs=10'. Values parsed as YAML so
    types are preserved (int/float/bool)."""
    for item in overrides or []:
        if "=" not in item:
            raise ConfigError(f"Override '{item}' must look like key.path=value")
        dotted, raw = item.split("=", 1)
        value = yaml.safe_load(raw)
        node: Any = cfg
        parts = dotted.split(".")
        for p in parts[:-1]:
            if not hasattr(node, p):
                raise ConfigError(f"Unknown override segment '{p}' in '{dotted}'")
            node = getattr(node, p)
        leaf = parts[-1]
        if not hasattr(node, leaf):
            raise ConfigError(f"Unknown override leaf '{leaf}' in '{dotted}'")
        setattr(node, leaf, value)


def load_config(path: str | Path, overrides: List[str] | None = None) -> Config:
    """Load and validate a YAML config into a typed `Config`."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"Failed to parse YAML config {path}: {e}") from e

    cfg = Config()
    _merge_into(cfg, raw)
    _apply_dotted_overrides(cfg, overrides or [])
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    if cfg.dataloader.batch_size < 1:
        raise ConfigError("dataloader.batch_size must be >= 1")
    if not (0.0 <= cfg.data.val_fraction < 1.0):
        raise ConfigError("data.val_fraction must be in [0, 1)")
    if cfg.model.decoder.backend not in {"local_gemma", "stub"}:
        raise ConfigError("model.decoder.backend must be 'local_gemma' or 'stub'")
    if cfg.model.vision.backend not in {"scratch_vit", "local_siglip"}:
        raise ConfigError("model.vision.backend must be 'scratch_vit' or 'local_siglip'")
    if cfg.model.vision.image_size % cfg.model.vision.patch_size != 0:
        raise ConfigError("vision.image_size must be divisible by vision.patch_size")
    if cfg.training.grad_accum_steps < 1:
        raise ConfigError("training.grad_accum_steps must be >= 1")


def clone(cfg: Config) -> Config:
    return copy.deepcopy(cfg)
