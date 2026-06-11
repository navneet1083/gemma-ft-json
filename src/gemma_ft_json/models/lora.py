"""Minimal, dependency-free LoRA (no `peft` -> nothing version-fragile,
nothing downloaded, and every line is auditable when debugging).

LoRA recap: a frozen weight W in R^{out x in} is augmented with a low-rank
update  dW = (alpha / r) * B @ A,  A in R^{r x in}, B in R^{out x r}.
Forward:  y = x @ W^T  +  (alpha/r) * ((x @ A^T) @ B^T)
   - x @ A^T : [*, in] @ [in, r]  -> [*, r]   (down-projection matmul)
   - (.) @ B^T: [*, r] @ [r, out] -> [*, out] (up-projection matmul)
B is zero-init => dW = 0 at step 0 => the model starts EXACTLY at the
pretrained function: stability by construction.
"""
from __future__ import annotations

from typing import Dict, Iterable, List

import torch
import torch.nn as nn

from ..exceptions import ModelLoadError


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 16, alpha: int = 32,
                 dropout: float = 0.05):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise ModelLoadError(f"LoRA target must be nn.Linear, got {type(base)}")
        self.base = base
        for p in self.base.parameters():            # the pretrained path is frozen
            p.requires_grad_(False)
        self.r, self.scaling = r, alpha / r
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)   # A: random
        # B stays zeros -> identity start (see module docstring).

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Frozen path matmul: x @ W^T (+bias)            [*, in] -> [*, out]
        y = self.base(x)
        # Low-rank path: ((drop(x) @ A^T) @ B^T) * s     two skinny matmuls
        lora = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return y + lora * self.scaling


def inject_lora(model: nn.Module, target_modules: Iterable[str],
                r: int, alpha: int, dropout: float) -> List[str]:
    """Replace every nn.Linear whose attribute name is in `target_modules`
    (e.g. q_proj / v_proj / o_proj inside Gemma attention) with LoRALinear.
    Returns the list of replaced module paths (for logging / sanity)."""
    targets = set(target_modules)
    replaced: List[str] = []
    for name, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            if child_name in targets and isinstance(child, nn.Linear):
                setattr(module, child_name, LoRALinear(child, r, alpha, dropout))
                replaced.append(f"{name}.{child_name}")
    if not replaced:
        raise ModelLoadError(
            f"inject_lora found no Linear named {sorted(targets)} — wrong "
            "target names for this architecture? Inspect model.named_modules().")
    return replaced


def mark_only_lora_trainable(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        p.requires_grad_("lora_A" in n or "lora_B" in n)


def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Adapter-only state dict (~2-6 MB) — what gets checkpointed for the LM."""
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if "lora_A" in k or "lora_B" in k}
