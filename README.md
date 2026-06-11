# gemma-ft-json

**Offline image-to-JSON table extraction by grafting a from-scratch vision encoder onto Gemma 3 270M.**

Gemma 3 270M is a *text-only* language model. It has no vision tower, no image
processor, and no pathway from pixels to its embedding space. This is the root
cause of the classic failed experiment: *"I applied LoRA and trained on
(image, JSON) pairs but the model never learned anything from the image."*
LoRA only perturbs existing weight matrices — it cannot create a sensory
modality the model never had. There is literally **no gradient path from the
pixels to the loss** unless you build one.

This repository builds that path, fully offline (no Hugging Face downloads),
sized for a Mac M4 with 48 GB unified memory (MPS backend).

---

## 1. Why the naive LoRA attempt fails (diagnosis)

1. **No input pathway.** A text-only decoder consumes token ids. An image is
   not a token id. Pasting base64/pixel values as text destroys spatial
   structure and explodes sequence length.
2. **Embedding scale mismatch** (the subtle killer). Gemma multiplies its
   token embeddings by `sqrt(hidden_size) ≈ 25.3` before the first decoder
   block. A randomly initialised projection layer emits vectors ~25× too
   small. The frozen LLM treats them as near-zero noise, attention ignores
   them, gradients through them are tiny, and "training" silently degenerates
   into the LLM memorising JSON priors while ignoring the image.
3. **Frozen tokenizer/vocabulary** means structure tokens (`{`, `"`, `:`)
   dominate the loss; content cells (the part the model must *read from the
   image*) are under-weighted.

The fixes implemented here, in order: a trainable vision encoder + projector
(creates the gradient path), **RMS calibration** of the projector output to
the empirical scale of real Gemma text embeddings (fixes the scale mismatch),
and a **structure-weighted loss** (punctuation 0.6×, cell content 1.4×).

## 2. Approach (LLaVA-style bridge + Donut-style data, minimised)

Informed by the literature: LLaVA showed a small trainable MLP projector can
bridge a vision encoder into a frozen LLM's embedding space; Donut showed
OCR-free document→JSON works well when trained on synthetically rendered
documents with pixel-perfect labels and a "read the document" pre-task.

We combine both, but replace LLaVA's downloaded CLIP tower (forbidden here)
with a small from-scratch encoder trained jointly — viable because the domain
is narrow (rendered tables), not open-world photos.

```
PNG 448×448 ─► TableViT-Lite (~11M, from scratch)
                conv stem /16 ─► 28×28 patches ─► 6 transformer blocks
                ─► 2×2 pixel-shuffle merge ─► 196 tokens × 1024-d
            ─► RMS-calibrated MLP projector ─► 196 "soft visual tokens" (640-d)
            ─► spliced after [BOS]:  [BOS][196 visual][prompt][JSON target]
            ─► frozen Gemma 3 270M (+ LoRA r=16 on q/v/o_proj)
            ─► structure-weighted causal-LM cross-entropy
```

Trainable parameters: vision encoder + projector + LoRA ≈ **15–20M** (~7% of
total). Checkpoints store *only* these (~50 MB), never the frozen base.

### Three-stage curriculum (set `training.stage` in config)
| Stage | Task | What it teaches |
|---|---|---|
| `read` | transcribe all visible text | grounds the encoder in pixels (Donut's pseudo-OCR) |
| `linearize` | emit `<row> a | b </row>` markup | row/column geometry, spans |
| `sft` | emit strict JSON | final target format; LoRA enabled here |

You can also run `sft` directly — the curriculum buys faster convergence and
better cell-content fidelity, not feasibility.

### Synthetic data with pixel-perfect labels
`data/synthetic_tables.py` renders tables with PIL: merged headers, row
spans, borderless layouts, zebra striping, currency/percent/date cells,
varying fonts and rules. Because we render them, the JSON ground truth is
exact — no OCR noise in the labels. Train on 4 000 / validate on 400 by
default (config-driven).

## 3. Repository layout

```
gemma-ft-json/
├── configs/config.yaml          # every path, hparam, probability — single source of truth
├── src/gemma_ft_json/
│   ├── exceptions.py            # typed errors (ConfigError, ShapeError, NumericalError, …)
│   ├── config/loader.py         # validated YAML → dataclasses, rejects unknown keys
│   ├── utils/                   # device (mps→cuda→cpu), seeding, JSONL metrics, registry
│   ├── data/                    # synthetic generator, Dataset, dynamic-padding collator
│   ├── models/
│   │   ├── vision_encoder.py    # TableViT-Lite (registered "table_vit_lite")
│   │   ├── projector.py         # RMS-calibrated MLP (registered "rms_mlp")
│   │   ├── lora.py              # dependency-free LoRA (no peft download)
│   │   ├── gemma_loader.py      # strictly local_files_only load
│   │   ├── vlm.py               # GemmaVisionForJSON: fusion, forward, generate
│   │   └── guards.py            # assert_shape / check_finite used at every boundary
│   ├── training/                # weighted loss, Trainer (resume, NaN policy), checkpoints
│   └── inference/predictor.py   # image → {raw_text, json, strictly_valid} + JSON repair
├── notebooks/  01 build dataset · 02 dataloader · 03 EDA · 04 training
│               05 live loss plots (tail metrics.jsonl while training) · 06 inference
├── scripts/    build_dataset.py · train.py (CLI twins of notebooks 01/04)
└── docs/architecture.svg
```

**Plug-and-play:** encoders and projectors register themselves in
`utils/registry.py`; to try a new encoder, add a file, decorate the class,
and change one YAML string. The collator/dataset/trainer are agnostic to it.

## 4. Getting the weights without Hugging Face downloads

The code sets `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` *before*
importing `transformers` and loads with `local_files_only=True`. Nothing is
fetched at runtime. Obtain Gemma 3 270M weights once through an allowed
channel — Kaggle Models ("gemma-3", Transformers format) or ai.google.dev —
copy the folder (config.json, tokenizer files, *.safetensors) to e.g.
`~/models/gemma-3-270m`, and point `paths.gemma_model_dir` at it. The
`transformers` *library* itself is just a pip package (PyPI), which is not a
model download.

## 5. Quick start

```bash
pip install -e .                       # torch, transformers, pyyaml, pillow, tqdm, matplotlib
# 1. edit configs/config.yaml → paths.gemma_model_dir
python scripts/build_dataset.py        # or notebook 01
jupyter lab notebooks/04_training.ipynb
# while training, open 05_live_plots.ipynb — it tails logs/metrics.jsonl live
```

Training behaviour (notebook 04 / `training/trainer.py`):
train/val split from the manifests, tqdm with loss/lr/grad-norm/skip-count,
AdamW with **two learning rates** (projector+encoder 1e-3, LoRA 1e-4), warmup
+ cosine schedule, gradient accumulation, `last.pt` every epoch, `best.pt`
**only when validation loss improves**, full resume (optimizer, scheduler,
RNG state, counters) via `training.resume_from`.

**Never-crash numerics:** shape assertions at every module boundary; inputs
sanitised with `nan_to_num`; if a step's loss is non-finite the step is
*skipped and logged* (not crashed); non-finite grad norm → gradients zeroed
and logged; CE computed in fp32 even on MPS; gradient clipping at 1.0.

## 6. Limitations (honest ones)

- **Synthetic→real gap.** The model learns rendered tables. Photographed or
  scanned tables (skew, shadows, camera noise) need an augmentation pass or a
  real-data fine-tune stage; hooks exist in the generator config.
- **270M is small.** Expect strong structure fidelity and good cell reading
  on clean renders, but long tables (>~25 rows) press against
  `max_seq_len=768` and content accuracy degrades before structure does.
- **196 visual tokens ≈ 32×32 px per token at 448².** Very dense small-font
  tables lose legibility; raise `image_size` + tokens at memory cost.
- **Greedy JSON is not guaranteed valid.** `Predictor` reports
  `strictly_valid` and applies conservative bracket-balancing repair; a
  constrained decoder would be the principled upgrade.
- **MPS quirks.** fp32 by default (bf16 on MPS is op-dependent);
  `PYTORCH_ENABLE_MPS_FALLBACK=1` is set so unsupported ops fall back to CPU
  silently (slower, not wrong).
- **No multi-table pages / nested JSON beyond two header levels** in the
  current generator — extendable in `synthetic_tables.py`.

## 7. Reusing pieces elsewhere

`TableViTLite`, `RMSCalibratedProjector`, `LoRALinear/inject_lora`, the
NaN-guarded `Trainer`, and `CheckpointManager` have no cross-dependencies on
each other beyond tensors in/out — each is importable into another project
as-is. The fusion trick (`vlm.py:_fuse_embeddings`, splice-after-BOS with
label re-padding) works for any decoder-only LM that exposes
`inputs_embeds`.
