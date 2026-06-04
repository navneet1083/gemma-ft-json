# gemma-ft-json

Fine-tune **Gemma 3 270M** (text-only) as the decoder of a small Vision-Language
Model that reads **table / document images** and emits **structured JSON** — fully
**offline** (no Hugging Face downloads), modular, config-driven, and MPS-friendly.

```
image ─▶ letterbox ─▶ vision encoder (frozen) ─▶ projector (trained) ─┐
prompt ─▶ tokenizer ─▶ embeddings ────────────────────────────────────┤
                                                                       ▼
                              concat[soft tokens | text] ─▶ Gemma 270M + LoRA ─▶ JSON
```

See `docs/architecture/ARCHITECTURE.md` for the design and `architecture.svg` for
the diagram.

---

## Why this design (short version)

Gemma 3 270M never saw pixels, so we bolt a **frozen** vision encoder and a small
**trained projector** onto it (LLaVA-style) and adapt the decoder with **LoRA**.
The decoder brings a strong structured-text prior + tokenizer; LoRA on a frozen
base is the main defense against catastrophic forgetting. The loss is next-token
cross-entropy **shaped for faithful extraction**: prompt/visual positions are
masked, and content tokens are up-weighted over JSON punctuation. Decoding is
greedy because sampling amplifies hallucination. Full rationale in the docs.

## The Hugging Face restriction (hard requirement)

This package **never downloads** weights. At import it sets `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`. Two ways to run:

1. **Zero assets (default, runs anywhere).** Pure-PyTorch fallbacks:
   `FromScratchViT`, `TinyStubDecoder`, byte-level `ByteTokenizer`. Great for
   wiring up the pipeline and the EDA/dataloader/training mechanics.
2. **Real Gemma/SigLIP, loaded locally.** Put weights you already have on disk and
   set `model.decoder.backend: local_gemma` + `local_dir`. Loading uses
   `local_files_only=True`; a missing dir raises `WeightsNotFoundError` instead of
   fetching anything. Obtain Gemma weights through your own approved/offline channel
   and place the directory locally — the code only ever *reads* it.

Verify the guard anytime: `python scripts/verify_offline.py --config configs/default.yaml`.

## Install

```bash
cd gemma-ft-json
python -m venv .venv && source .venv/bin/activate
pip install -e .                 # core
pip install -e ".[hf]"           # + local transformers loading (Gemma/SigLIP)
pip install -e ".[curation]"     # + PyMuPDF/Tesseract grounding (optional)
pip install -e ".[serve,dev]"    # + FastAPI service, pytest, notebooks
```

Requires Python ≥ 3.10 and PyTorch ≥ 2.2 (Apple-Silicon MPS auto-detected).

## Configure

Everything lives in `configs/default.yaml` — dataset paths, model backends, LoRA,
optimizer, training schedule, logging. Set at least:

```yaml
paths:
  images_dir: "/abs/path/to/images"
  json_dir:   "/abs/path/to/json"   # one <stem>.json per <stem>.<img>
```

Override on the CLI: `--set training.epochs=10 dataloader.batch_size=4`.

## Run

```bash
# 1) verify offline guard
python scripts/verify_offline.py --config configs/default.yaml
# 2) train (build manifest + split + fit). Resumes from runs/<name>/last.ckpt.
python scripts/train.py --config configs/default.yaml
# 3) predict on one image with the best checkpoint
python scripts/predict.py \
  --snapshot runs/gemma-ft-json/config.snapshot.yaml \
  --checkpoint runs/gemma-ft-json/best.ckpt \
  --image /abs/path/to/table.png
# 4) optional deployment service
python scripts/serve.py --config configs/deploy.yaml
```

Run the test suite: `pytest -q` (uses the tiny stub backend; no downloads).

## Notebooks (`notebooks/`)

1. **01_build_dataset** — build `train/val.jsonl` from images+JSON; each curation
   utility exposed and demoed.
2. **02_dataloader** — tokenizer, transform, `TableJsonDataset`, padded loaders,
   batch shapes.
3. **03_eda** — sample image + JSON, target-length histogram.
4. **04_train** — descriptive training: train/val split, **resume**, tqdm with
   loss/lr/grad-norm, JSONL metrics, **save-best-only**, per-epoch checkpoints.
5. **05_plots** — read `runs/<name>/metrics.jsonl` and plot loss/lr/grad-norm;
   re-run mid-training for **live** curves.
6. **06_inference_test** — load `best.ckpt`, pick an image, show predicted JSON.

## Project layout

```
gemma-ft-json/
├── configs/            default.yaml (master), deploy.yaml
├── docs/architecture/  architecture.svg + ARCHITECTURE.md
├── notebooks/          01..06 (see above)
├── scripts/            train.py, predict.py, verify_offline.py, serve.py
├── src/gemma_ft_json/
│   ├── config.py exceptions.py tokenization.py scripts_entry.py
│   ├── utils/   device · seed · logging_utils · checks
│   ├── data/    transforms · build_dataset · dataset · collate
│   ├── models/  vision_encoder · projector · backends(+LoRA) · losses · vlm
│   ├── training/ optim · checkpoint · trainer
│   └── inference/ predictor
└── tests/              test_smoke.py
```

## Training behavior

Resume is automatic (`runs/<name>/last.ckpt`). Per step we log loss, LR, and
grad-norm to console (tqdm) and to `metrics.jsonl`. Validation runs on a cadence
(`eval_every_steps`, or once per epoch if `0`); `best.ckpt` is written **only when
val loss improves** (`save_best_only`). Checkpoints store *trainable* params only
(projector + LoRA) plus optimizer/scheduler/epoch/step — the frozen base reloads
from its local dir. Gradients are clipped; NaN/Inf is caught at the source.

## Limitations (be honest)

* **270M is small.** Expect strong structure but limited capacity on dense,
  multi-page, or visually noisy tables; treat the stub-only runs as wiring demos,
  not accuracy benchmarks.
* **Byte tokenizer is a fallback.** It runs without assets but is far weaker than
  the real Gemma tokenizer; use `local_gemma` for real results.
* **Grounding & RL are scaffolding.** `<loc>` tokens, bbox supervision, and RL with
  exact-match/TEDS rewards have hooks but are not fully wired end-to-end.
* **Random split.** `split_manifest` shuffles records; for a true generalization
  test, split by document source so eval layouts are unseen.
* **Reference generate is slow.** The backend-agnostic loop recomputes the full
  sequence per token (no KV cache); use the local HF model's cached `.generate`
  for production speed.
* **No schema enforcement by default.** Output is greedy free-form JSON; add
  schema-constrained decoding / validation for guarantees.

## License

Apache-2.0. You are responsible for complying with the licenses/terms of any model
weights (e.g. Gemma) you load locally.
