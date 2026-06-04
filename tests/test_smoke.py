"""End-to-end smoke tests on a TINY stub config (no downloads, CPU-friendly).

Verifies the pieces the spec cares about most:
  * the model wires together and the forward pass runs,
  * the loss is finite and `.backward()` produces finite gradients (NaN guard),
  * greedy `generate` returns tokens,
  * the dataset/collator emit correctly-shaped, mask-aligned batches.
"""
import torch

from gemma_ft_json.config import Config
from gemma_ft_json.models.vlm import build_model
from gemma_ft_json.data.collate import Collator
from gemma_ft_json.data.dataset import IGNORE_INDEX


def _tiny_cfg() -> Config:
    cfg = Config()
    # Stub decoder + from-scratch ViT at tiny dims so the test is fast on CPU.
    cfg.model.decoder.backend = "stub"
    cfg.model.decoder.hidden_size = 32
    cfg.model.decoder.max_seq_len = 128
    cfg.model.decoder.n_structure_tokens = 4
    cfg.model.vision.backend = "scratch_vit"
    cfg.model.vision.image_size = 64
    cfg.model.vision.patch_size = 16
    cfg.model.vision.embed_dim = 32
    cfg.model.vision.depth = 1
    cfg.model.vision.num_heads = 2
    cfg.model.vision.freeze = True
    cfg.model.lora.enabled = True
    cfg.model.lora.rank = 4
    cfg.model.lora.target_modules = ["q_proj", "v_proj"]
    return cfg


def test_forward_backward_no_nan():
    cfg = _tiny_cfg()
    model, tok = build_model(cfg)
    model.train()

    B, S = 2, cfg.model.vision.image_size
    pixel_values = torch.randn(B, 3, S, S)
    ids = tok.encode("hello", add_bos=True, add_eos=False) + tok.encode('{"a":1}', add_eos=True)
    input_ids = torch.tensor([ids, ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[:, :3] = IGNORE_INDEX  # mask a fake prompt region

    out = model(pixel_values=pixel_values, input_ids=input_ids,
                attention_mask=attention_mask, labels=labels)
    loss = out["loss"]
    assert torch.isfinite(loss).all(), "loss is not finite"

    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads, "no gradients flowed to trainable params"
    for g in grads:
        assert torch.isfinite(g).all(), "non-finite gradient detected"


def test_generate_runs():
    cfg = _tiny_cfg()
    model, tok = build_model(cfg)
    pixel_values = torch.randn(1, 3, cfg.model.vision.image_size, cfg.model.vision.image_size)
    prompt_ids = torch.tensor(tok.encode("x", add_bos=True), dtype=torch.long)
    out = model.generate(pixel_values, prompt_ids, max_new_tokens=5, eos_id=tok.eos_id)
    assert isinstance(out, list)


def test_collator_shapes_and_masking():
    cfg = _tiny_cfg()
    _, tok = build_model(cfg)
    collate = Collator(pad_id=tok.pad_id)
    items = []
    for n in (5, 8):  # different lengths -> exercise padding
        ids = list(range(n))
        items.append({
            "pixel_values": torch.randn(3, 64, 64),
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(ids, dtype=torch.long),
            "length": torch.tensor(n),
        })
    batch = collate(items)
    assert batch["input_ids"].shape == (2, 8)
    assert batch["attention_mask"].shape == (2, 8)
    # Shorter sample's padded tail must be masked out.
    assert batch["attention_mask"][0, 5:].sum().item() == 0
    assert batch["pixel_values"].shape == (2, 3, 64, 64)
