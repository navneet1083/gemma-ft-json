"""The assembled vision-language model + build factory.

Forward (LLaVA-style):
    image -> encoder -> patch embeds [B,N,Dv] -> projector -> soft tokens [B,N,H]
    text  -> decoder.embed -> text embeds [B,T,H]
    concat([soft ; text]) -> decoder -> logits [B,N+T,V]
    masked, shifted CE on the JSON target -> loss

Every junction is dimension-checked and finiteness-guarded so a mis-wire fails at
the boundary, not as a downstream NaN. The encoder/projector/decoder are swappable
via config -> this is the plug-and-play "fusion" point.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from ..config import Config
from ..tokenization import BaseTokenizer, build_tokenizer
from ..utils.checks import check_shape, assert_finite
from .vision_encoder import build_vision_encoder
from .projector import build_projector
from .backends import build_decoder, inject_lora
from .losses import faithful_lm_loss, IGNORE_INDEX

_SCAFFOLD_CHARS = ["{", "}", "[", "]", ":", ",", '"']


class GemmaTableVLM(nn.Module):
    def __init__(self, encoder: nn.Module, projector: nn.Module, decoder: nn.Module,
                 loss_cfg, scaffold_ids: Optional[Set[int]] = None):
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.decoder = decoder
        self.loss_cfg = loss_cfg
        self.scaffold_ids = scaffold_ids or set()

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """image pixels -> soft visual tokens in the decoder's hidden space."""
        feats = self.encoder(pixel_values)                  # [B, N, Dv]
        check_shape(feats, (None, None, None), "encoder_feats")
        return self.projector(feats)                         # [B, N, H]

    def forward(self, pixel_values, input_ids, attention_mask, labels) -> Dict[str, torch.Tensor]:
        B = input_ids.size(0)
        soft = self.encode_image(pixel_values)               # [B, N, H]
        n_vis = soft.size(1)
        text_embeds = self.decoder.embed_tokens(input_ids)   # [B, T, H]

        full_embeds = torch.cat([soft, text_embeds], dim=1)  # [B, N+T, H]
        vis_mask = torch.ones(B, n_vis, dtype=attention_mask.dtype, device=attention_mask.device)
        full_mask = torch.cat([vis_mask, attention_mask], dim=1)
        vis_labels = torch.full((B, n_vis), IGNORE_INDEX, dtype=labels.dtype, device=labels.device)
        full_labels = torch.cat([vis_labels, labels], dim=1)

        logits = self.decoder(inputs_embeds=full_embeds, attention_mask=full_mask)
        assert_finite(logits, "vlm_logits")

        loss = faithful_lm_loss(
            logits, full_labels,
            content_token_weight=self.loss_cfg.content_token_weight,
            scaffold_ids=self.scaffold_ids,
            label_smoothing=self.loss_cfg.label_smoothing,
        )
        return {"loss": loss, "logits": logits, "n_vis": n_vis}

    @torch.no_grad()
    def generate(self, pixel_values, prompt_ids: torch.Tensor, max_new_tokens: int,
                 eos_id: int) -> List[int]:
        """Greedy decoding from image + prompt.

        Greedy (argmax), NOT sampling — sampling amplifies hallucination, the
        opposite of faithful extraction. This reference loop recomputes the full
        sequence each step (no KV cache) to stay backend-agnostic; for speed in
        production use the HF model's cached `.generate` on LocalGemmaDecoder.
        """
        self.eval()
        device = pixel_values.device
        soft = self.encode_image(pixel_values)                              # [1, N, H]
        prompt_embeds = self.decoder.embed_tokens(prompt_ids.unsqueeze(0))  # [1, P, H]
        cur = torch.cat([soft, prompt_embeds], dim=1)
        generated: List[int] = []
        for _ in range(max_new_tokens):
            mask = torch.ones(cur.size(0), cur.size(1), dtype=torch.long, device=device)
            logits = self.decoder(inputs_embeds=cur, attention_mask=mask)
            next_id = int(logits[:, -1, :].argmax(dim=-1).item())
            if next_id == eos_id:
                break
            generated.append(next_id)
            nxt = self.decoder.embed_tokens(torch.tensor([[next_id]], device=device, dtype=torch.long))
            cur = torch.cat([cur, nxt], dim=1)
        return generated

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_total(self) -> int:
        return sum(p.numel() for p in self.parameters())


def _scaffold_ids(tokenizer: BaseTokenizer) -> Set[int]:
    """Token ids for JSON scaffolding symbols (to up-weight content tokens). Keep
    only single-token encodings -> simple and safe heuristic."""
    ids: Set[int] = set()
    for ch in _SCAFFOLD_CHARS:
        enc = tokenizer.encode(ch, add_bos=False, add_eos=False)
        if len(enc) == 1:
            ids.add(enc[0])
    return ids


def build_model(cfg: Config) -> Tuple[GemmaTableVLM, BaseTokenizer]:
    """Wire the full model from config. Returns (model, tokenizer).

    Order matters: tokenizer first (sets vocab incl. added tokens) -> decoder
    sized to that vocab -> encoder -> projector bridging encoder dim to hidden
    size -> optional LoRA on the (frozen-base) decoder.
    """
    tokenizer = build_tokenizer(cfg.model.decoder)
    decoder = build_decoder(cfg.model.decoder, tokenizer)
    encoder = build_vision_encoder(cfg.model.vision)

    enc_dim = getattr(encoder, "embed_dim", None) or cfg.model.vision.embed_dim
    projector = build_projector(cfg.model.projector, in_dim=enc_dim, out_dim=decoder.hidden_size)

    if cfg.model.lora.enabled:
        n = inject_lora(decoder, cfg.model.lora.target_modules, rank=cfg.model.lora.rank,
                        alpha=cfg.model.lora.alpha, dropout=cfg.model.lora.dropout)
        setattr(decoder, "_lora_layers_adapted", n)  # surfaced in logs

    model = GemmaTableVLM(encoder=encoder, projector=projector, decoder=decoder,
                          loss_cfg=cfg.loss, scaffold_ids=_scaffold_ids(tokenizer))
    return model, tokenizer
