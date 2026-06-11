"""GemmaVisionForJSON: the fused vision-language model.

Token layout fed to Gemma (soft visual prefix, LLaVA-style):

    [BOS] [v_1 ... v_196] [prompt tokens ...] [target tokens ...]
           ^^^^^^^^^^^^^
           projected visual embeddings — *soft tokens*: continuous vectors
           injected via `inputs_embeds`, they never exist in the vocabulary.

Why prefix injection instead of adding an <image> token to the vocab?
Gemma 3's 262k-row embedding table is weight-tied to the LM head; resizing
it would (a) perturb the output distribution and (b) add 262k * 640 new
logit rows of optimizer state. Prefix injection touches neither.

Trainable parameters by stage:
    align : vision encoder + projector            (Gemma 100% frozen)
    sft   : vision encoder + projector + LoRA(q,v,o)
Everything else in Gemma stays frozen — total trainables ~13M, trivially
inside the M4's 48 GB even with Adam moments in fp32.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from ..config.loader import ModelCfg
from ..exceptions import ModelLoadError
from .guards import assert_shape, check_finite
from .lora import inject_lora, lora_state_dict
from .projector import PROJECTORS
from .vision_encoder import VISION_ENCODERS


class GemmaVisionForJSON(nn.Module):
    def __init__(self, gemma: nn.Module, tokenizer, model_cfg: ModelCfg,
                 image_size: int, stage: str = "sft",
                 vision_name: str = "table_vit_lite",
                 projector_name: str = "rms_mlp"):
        super().__init__()
        self.gemma = gemma
        self.tok = tokenizer
        self.stage = stage
        self.lm_dim = int(gemma.config.hidden_size)        # 640 for Gemma3-270M

        # ---- plug-and-play sub-modules resolved from registries ----
        v = model_cfg.vision
        self.vision = VISION_ENCODERS.get(vision_name)(
            image_size=image_size, embed_dim=v.embed_dim, depth=v.depth,
            num_heads=v.num_heads, patch_stride=v.patch_stride,
            merge_factor=v.merge_factor, drop_rate=v.drop_rate)
        self.projector = PROJECTORS.get(projector_name)(
            in_dim=self.vision.out_dim, lm_dim=self.lm_dim,
            hidden_mult=model_cfg.projector.hidden_mult,
            rms_calibrate=model_cfg.projector.rms_calibrate)
        self.num_visual_tokens = self.vision.num_tokens

        # ---- freeze Gemma; optionally inject LoRA for the sft stage ----
        for p in self.gemma.parameters():
            p.requires_grad_(False)
        self.lora_paths = []
        if stage == "sft":
            self.lora_paths = inject_lora(
                self.gemma, model_cfg.lora.target_modules,
                model_cfg.lora.r, model_cfg.lora.alpha, model_cfg.lora.dropout)

        # ---- calibrate projector output RMS against real text embeddings ----
        # (See projector.py: THE fix for visual tokens being "whispers".)
        if model_cfg.projector.rms_calibrate:
            sample = self.tok("The quick brown fox reads tables.",
                              return_tensors="pt")["input_ids"]
            rms = self.projector.calibrate(self.gemma.get_input_embeddings(), sample)
            if not (0.01 < rms < 1e4):
                raise ModelLoadError(f"Embedding RMS calibration insane: {rms}")

    # ------------------------------------------------------------------ #
    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def param_groups(self, lr_projector: float, lr_lora: float):
        """Two LR groups: random-init modules (fast) vs LoRA (slow).
        A single LR either starves the projector or fries the LoRA."""
        proj_vision, lora = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (lora if ("lora_A" in n or "lora_B" in n) else proj_vision).append(p)
        groups = [{"params": proj_vision, "lr": lr_projector, "name": "vision+projector"}]
        if lora:
            groups.append({"params": lora, "lr": lr_lora, "name": "lora"})
        return groups

    # ------------------------------------------------------------------ #
    def _fuse_embeddings(self, pixel_values: torch.Tensor,
                         input_ids: torch.Tensor,
                         attention_mask: torch.Tensor):
        """Build [BOS][visual][text...] inputs_embeds + extended mask."""
        B = input_ids.shape[0]
        embed = self.gemma.get_input_embeddings()

        # Text path: embedding lookup == one-hot(input_ids) @ E, realized as a
        # gather over the [vocab, lm_dim] embedding matrix.
        text_embeds = embed(input_ids)                       # [B, T_txt, D]

        vis_tokens = self.vision(pixel_values)               # [B, 196, 4*Dv]
        vis_embeds = self.projector(vis_tokens)              # [B, 196, D]
        vis_embeds = vis_embeds.to(text_embeds.dtype)
        assert_shape(vis_embeds, (B, self.num_visual_tokens, self.lm_dim),
                     "visual_embeds")

        # Splice AFTER the BOS token (position 0): Gemma's distribution is
        # anchored on BOS-first sequences; keeping it first costs nothing.
        fused = torch.cat([text_embeds[:, :1], vis_embeds, text_embeds[:, 1:]], dim=1)
        vis_mask = torch.ones(B, self.num_visual_tokens,
                              dtype=attention_mask.dtype, device=attention_mask.device)
        fused_mask = torch.cat(
            [attention_mask[:, :1], vis_mask, attention_mask[:, 1:]], dim=1)
        return fused, fused_mask

    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Returns logits aligned to the FUSED sequence; labels (if given) are
        re-padded here with -100 over the visual block so the trainer's loss
        never scores visual positions."""
        fused, fused_mask = self._fuse_embeddings(
            pixel_values, input_ids, attention_mask)

        out = self.gemma(inputs_embeds=fused, attention_mask=fused_mask)
        # lm_head logits = hidden @ E^T (weight-tied):
        #   [B, T, 640] @ [640, 262k] -> [B, T, 262k]
        logits = check_finite(out.logits, "lm_logits", sanitize=True)

        fused_labels = None
        if labels is not None:
            B = labels.shape[0]
            pad = torch.full((B, self.num_visual_tokens), -100,
                             dtype=labels.dtype, device=labels.device)
            fused_labels = torch.cat([labels[:, :1], pad, labels[:, 1:]], dim=1)

        return {"logits": logits, "labels": fused_labels,
                "attention_mask": fused_mask}

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate_from_image(self, pixel_values: torch.Tensor,
                            prompt_ids: torch.Tensor, max_new_tokens: int = 512,
                            temperature: float = 0.0,
                            eot_id: Optional[int] = None) -> torch.Tensor:
        """Greedy/temperature decoding with a manual KV-cache-free loop.

        WHY manual instead of model.generate()? generate() with
        inputs_embeds + custom prefixes differs subtly across transformers
        versions; for a 270M model on M4, recomputing the (short) sequence
        per step is fast enough and 100% version-proof.
        """
        self.eval()
        attn = torch.ones_like(prompt_ids)
        fused, fused_mask = self._fuse_embeddings(pixel_values, prompt_ids, attn)
        embed = self.gemma.get_input_embeddings()
        generated = []
        eos = self.tok.eos_token_id

        for _ in range(max_new_tokens):
            out = self.gemma(inputs_embeds=fused, attention_mask=fused_mask)
            next_logits = out.logits[:, -1, :]               # [B, vocab]
            next_logits = torch.nan_to_num(next_logits, nan=-1e9)
            if temperature and temperature > 0:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                nxt = torch.multinomial(probs, 1)
            else:
                nxt = next_logits.argmax(dim=-1, keepdim=True)
            tid = int(nxt[0, 0])
            if tid == eos or (eot_id is not None and tid == eot_id):
                break
            generated.append(tid)
            fused = torch.cat([fused, embed(nxt)], dim=1)
            fused_mask = torch.cat(
                [fused_mask, torch.ones_like(fused_mask[:, :1])], dim=1)
        return torch.tensor(generated, dtype=torch.long)

    # ------------------------------------------------------------------ #
    def trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        """Vision + projector + LoRA only (~50 MB) — Gemma base weights are
        immutable and reproducible from the local model dir, so checkpoints
        stay small and resume stays fast."""
        sd = {f"vision.{k}": v.detach().cpu()
              for k, v in self.vision.state_dict().items()}
        sd.update({f"projector.{k}": v.detach().cpu()
                   for k, v in self.projector.state_dict().items()})
        sd.update({f"gemma_lora.{k}": v
                   for k, v in lora_state_dict(self.gemma).items()})
        return sd

    def load_trainable_state_dict(self, sd: Dict[str, torch.Tensor]) -> None:
        vis = {k[len("vision."):]: v for k, v in sd.items() if k.startswith("vision.")}
        proj = {k[len("projector."):]: v for k, v in sd.items()
                if k.startswith("projector.")}
        lora = {k[len("gemma_lora."):]: v for k, v in sd.items()
                if k.startswith("gemma_lora.")}
        self.vision.load_state_dict(vis, strict=True)
        self.projector.load_state_dict(proj, strict=True)
        if lora:
            missing, unexpected = self.gemma.load_state_dict(lora, strict=False)
            if unexpected:
                raise ModelLoadError(f"Unexpected LoRA keys: {unexpected[:5]}...")
