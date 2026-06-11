"""YAML config loading with validation and path expansion.

Why a dataclass-backed loader instead of passing dicts around?
Typos in nested dict keys fail *silently* deep inside training; the
dataclass surface fails *loudly* at startup, which is where you want it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..exceptions import ConfigError


def _expand(p: str) -> str:
    """Expand ~ and make relative paths absolute w.r.t. the repo root."""
    return str(Path(os.path.expanduser(p)).resolve())


@dataclass
class PathsCfg:
    gemma_model_dir: str
    data_root: str
    raw_images_dir: str
    processed_dir: str
    manifest_train: str
    manifest_val: str
    checkpoints_dir: str
    logs_dir: str
    log_file: str
    metrics_file: str


@dataclass
class DeviceCfg:
    preferred: str = "auto"
    dtype: str = "float32"
    mps_fallback_env: bool = True


@dataclass
class DatasetCfg:
    num_train_samples: int = 4000
    num_val_samples: int = 400
    image_size: int = 448
    min_rows: int = 2
    max_rows: int = 9
    min_cols: int = 2
    max_cols: int = 6
    p_merged_header: float = 0.35
    p_row_span: float = 0.20
    p_no_gridlines: float = 0.25
    p_zebra: float = 0.30
    seed: int = 1337


@dataclass
class VisionCfg:
    embed_dim: int = 256
    depth: int = 6
    num_heads: int = 4
    patch_stride: int = 16
    merge_factor: int = 2
    drop_rate: float = 0.0


@dataclass
class ProjectorCfg:
    hidden_mult: int = 2
    rms_calibrate: bool = True


@dataclass
class LoraCfg:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])


@dataclass
class ModelCfg:
    vision: VisionCfg = field(default_factory=VisionCfg)
    projector: ProjectorCfg = field(default_factory=ProjectorCfg)
    lora: LoraCfg = field(default_factory=LoraCfg)
    max_seq_len: int = 768


@dataclass
class TrainingCfg:
    stage: str = "sft"
    epochs: int = 10
    batch_size: int = 4
    grad_accum_steps: int = 4
    lr_projector: float = 1e-3
    lr_lora: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    scheduler: str = "cosine"
    grad_clip_norm: float = 1.0
    label_smoothing: float = 0.0
    structure_loss_weight: float = 0.6
    content_loss_weight: float = 1.4
    val_every_epochs: int = 1
    log_every_steps: int = 10
    save_best_only: bool = True
    resume_from: Optional[str] = None
    num_workers: int = 2
    seed: int = 1337


@dataclass
class InferenceCfg:
    max_new_tokens: int = 512
    temperature: float = 0.0
    json_guard: bool = True


@dataclass
class AppConfig:
    paths: PathsCfg
    device: DeviceCfg
    dataset: DatasetCfg
    model: ModelCfg
    training: TrainingCfg
    inference: InferenceCfg
    raw: Dict[str, Any] = field(default_factory=dict)


def _build(dc_cls, data: Dict[str, Any]):
    """Build a dataclass from a dict, raising ConfigError for unknown keys."""
    valid = {f for f in dc_cls.__dataclass_fields__}
    unknown = set(data) - valid
    if unknown:
        raise ConfigError(f"Unknown keys for {dc_cls.__name__}: {sorted(unknown)}")
    return dc_cls(**data)


def load_config(path: str = "configs/config.yaml") -> AppConfig:
    """Load + validate the YAML config. Single entry-point for all modules."""
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"Config file not found: {p.resolve()}")
    try:
        with open(p, "r") as fh:
            raw = yaml.safe_load(fh)
    except yaml.YAMLError as exc:  # malformed YAML -> typed, actionable error
        raise ConfigError(f"Invalid YAML in {p}: {exc}") from exc

    try:
        paths = _build(PathsCfg, {k: _expand(v) for k, v in raw["paths"].items()})
        mc = raw.get("model", {})
        model = ModelCfg(
            vision=_build(VisionCfg, mc.get("vision", {})),
            projector=_build(ProjectorCfg, mc.get("projector", {})),
            lora=_build(LoraCfg, mc.get("lora", {})),
            max_seq_len=mc.get("max_seq_len", 768),
        )
        cfg = AppConfig(
            paths=paths,
            device=_build(DeviceCfg, raw.get("device", {})),
            dataset=_build(DatasetCfg, raw.get("dataset", {})),
            model=model,
            training=_build(TrainingCfg, raw.get("training", {})),
            inference=_build(InferenceCfg, raw.get("inference", {})),
            raw=raw,
        )
    except KeyError as exc:
        raise ConfigError(f"Missing required config section: {exc}") from exc

    # Create output dirs eagerly so later code never hits FileNotFoundError.
    for d in (cfg.paths.data_root, cfg.paths.raw_images_dir, cfg.paths.processed_dir,
              cfg.paths.checkpoints_dir, cfg.paths.logs_dir):
        Path(d).mkdir(parents=True, exist_ok=True)
    return cfg
