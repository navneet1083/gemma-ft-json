# PharmaLLM: A Proposal to Build an In-House Pharmacy LLM From Scratch
### Research dossier and slide-ready material for a 10–15 slide leadership pitch

## TL;DR
- **Build a domain-specialized 3B–13B pharmacy LLM in stages — start with continued pre-training + post-training of an open-weight base, and escalate to a from-scratch model only after a pilot proves value.** The precedent is strong and repeatable: BloombergGPT (50.6B, 709B tokens), GatorTronGPT (20B, 277B words of clinical text), Meditron-70B (48.1B-token medical adaptation of Llama-2), and BioMedLM (2.7B, beat much larger models on MedQA) all show that *domain data beats raw scale* for niche tasks.
- **Frontier models (GPT-4o, Claude, Gemini) are not safe or economical for pharmacy at our scale**: documented medication hallucinations (inappropriate, potentially harmful drug-interaction advice in 14% of GPT-4 responses in one 2024 study), PHI/HIPAA exposure from sending data to external APIs, zero knowledge of our proprietary formulary/claims/prior-auth data, poor tokenization of drug codes, and recurring per-token API costs that dwarf a one-time training run.
- **The economics work**: a 3B model on 300B tokens costs roughly **$9–11K** of raw H100 compute; a 13B model on 1T tokens roughly **$137–164K**. Against recurring frontier API fees (GPT-4o at $2.50 input / $10 output per 1M tokens; Claude 3.5 Sonnet at $3/$15), a one-time training asset plus cheap in-house inference pays back quickly *and* keeps all PHI inside our security perimeter.

## Key Findings

1. **Domain-specific models reliably beat general models on their turf, at a fraction of the size.** BioMedLM (2.7B, Stanford CRFM + MosaicML, arXiv:2403.18421) reached a state-of-the-art **50.3% on MedQA** trained only on PubMed — beating the similarly-sized general GPT-Neo 2.7B by ~17 points. Palmyra-Med-70B (Writer) averaged **85.9% across medical benchmarks in a zero-shot attempt, surpassing Med-PaLM-2 (84%, which required six attempts)** and GPT-4 base. PharmBERT demonstrated that FDA drug labels are a *distinct language domain* that generic, biomedical (BioBERT), and clinical (ClinicalBERT) models all handle sub-optimally.

2. **Frontier models make dangerous medication errors.** Sikora et al. (medRxiv, June 30 2024, doi:10.1101/2024.06.29.24309701) found that GPT-4 gave **inappropriate clinical recommendations potentially causing patient harm in 14% of responses (2/14)** and that **63% of responses were accurate but incomplete (19/30)**; GPT-4 also over-flagged relevance, categorizing 86% of interactions as highly clinically relevant versus experts' 53%. Medical hallucination is described in the literature as an intrinsic property of LLMs, and incorrect dosages or interactions can be life-threatening.

3. **The privacy/compliance case is decisive.** Sending pharmacy claims and patient medication data to external APIs creates HIPAA/PHI exposure. Amazon's own work is instructive: Sagar et al. ("Large language models for preventing medication direction errors in online pharmacies," *Nature Medicine* 2024, doi:10.1038/s41591-024-02933-8) built the purpose-built MEDIC system that **reduced near-miss dispensing events by 33% (CI 26–40%)**, while the two general LLM benchmarks produced **1.51× and 4.38× more near-miss events than MEDIC**. In-house models trained on de-identified data and run on our own infrastructure keep data inside our walls — the same rationale BioMedLM cites (runs on local hardware, no data sent over the internet).

4. **The modern recipe is well-established and reproducible.** Decoder-only Transformer with RMSNorm, RoPE, Grouped-Query Attention (GQA), and SwiGLU (the LLaMA-3 / Meditron consensus), FlashAttention, BF16 mixed precision, FSDP/Megatron parallelism, and cosine/WSD learning-rate schedules — followed by SFT + DPO post-training and RAG for up-to-date drug facts.

5. **Costs are modest and knowable.** Using the standard compute formula C ≈ 6ND FLOPs (Kaplan et al. 2020, arXiv:2001.08361; Hoffmann et al./Chinchilla 2022, arXiv:2203.15556), H100 at ~$2.50–3/GPU-hr and 40% Model FLOPs Utilization: **3B×300B ≈ $9–11K; 7B×500B ≈ $37–44K; 13B×1T ≈ $137–164K** of raw compute. Even DeepSeek-V3 (671B total/37B active, 14.8T tokens) trained for **2.788M H800 GPU-hours ≈ $5.576M** — an order of magnitude below GPT-4-class spend — proving that architecture and data curation, not brute spend, drive cost-effectiveness.

## Details

### 1. The Pharmacy Domain and Its Data
The pharmacy section spans prescription processing and e-prescribing (SIG code/direction parsing), drug-drug interaction (DDI) checking, medication therapy management, pharmacovigilance/adverse-event detection, formulary management, prior authorization, clinical decision support, drug-information queries, medication reconciliation, pharmacy claims adjudication, inventory, and compounding.

The data is uniquely structured and terminology-dense. Key standards:
- **RxNorm** (NLM) — normalized names for clinical drugs; links drug vocabularies via RxCUI identifiers; normalizes NDCs to 11-digit HIPAA format.
- **NDC** (National Drug Code) — 10/11-digit product identifier and the pharmacy industry's key product-level code, used for ordering, dispensing, billing, rebates, and adverse-event reporting. NCPDP has formally argued NDC cannot be replaced by RxNorm because RxNorm "lacks the specificity required to uniquely identify a product."
- **SNOMED CT** — the world's largest clinical terminology.
- **NCPDP standards** (SCRIPT for e-prescribing, Telecommunication Standard for claims) — HIPAA-mandated for pharmacy transactions.
- **FDA SPL** (Structured Product Labeling) — the drug-labeling format that formed PharmBERT's corpus.

This code-heavy, abbreviation-heavy text is exactly where a **custom domain tokenizer** helps. BioMedLM's biomedical tokenizer (28,896-token vocabulary) encodes "chromatography" as one token versus "chrom/atography" in GPT-2; GatorTron used a 50K custom clinical vocabulary; BloombergGPT used a 131,072-token vocabulary (double the standard ~50K) to capture tickers and numerical structures. A pharmacy tokenizer that treats NDC codes, RxCUIs, and SIG abbreviations as coherent tokens gains both efficiency and accuracy.

### 2. Why Frontier Models Fall Short for Pharmacy
- **Hallucinated drug facts/dosages.** The Sikora GPT-4 DDI study (above) found 14% harmful and 63% incomplete recommendations. Clinical note-generation studies report hallucination rates of 15–35% depending on model and task; a controlled clinical-summarization framework (npj Digital Medicine 2025) measured a 1.47% hallucination and 3.45% omission rate even in a tuned setting. Researchers note LLMs generate "a mix of real and fictional medications" because they don't consult a validated list — they predict statistically plausible text.
- **PHI/HIPAA risk** from transmitting patient data to third-party APIs.
- **No proprietary knowledge** of our formulary, claims history, prior-auth rules, or negotiated pricing.
- **Tokenization problems** with drug names and medical codes fragment meaning across subword tokens.
- **Cost at scale** — recurring per-token API fees (GPT-4o $2.50/$10; Claude 3.5 Sonnet $3/$15 per 1M tokens) compound with volume.
- **Knowledge cutoff** — new drug approvals and formulary changes aren't in a frozen model; ASHP explicitly warns that "anyone who uses ChatGPT for medication information should verify with trusted sources."
- **Cannot fine-tune deeply** on closed frontier APIs.

Notably, general models *can* pass pharmacy exams — GPT-4 scored **87% (McGraw Hill) and 83.5% (RxPrep) on two NAPLEX practice sets, and 96.1% on adverse-drug-reaction questions** (Angel et al., *Am. J. Pharm. Educ.* 2024) — but exam performance ≠ safe, grounded, proprietary-data-aware production behavior, and models degrade sharply on complex multi-step DDI reasoning and select-all questions (GPT-4 dropped to 73.1% on select-all).

### 3. Precedents — Domain Models Built From Scratch or Adapted

| Model | Params | Training data | Compute / cost | Outcome |
|---|---|---|---|---|
| **BloombergGPT** (finance, from scratch) | 50.6B | 709B tokens (363B proprietary FinPile + 345B public) | ~1.3M A100 GPU-hrs; 64 clusters × 8 A100 40GB (512 GPUs), 53 days, 139,200 steps; ~$2.67–10M | Beat similar open models on finance tasks by 8–10 pts; ~Chinchilla-optimal at 50B |
| **GatorTronGPT** (clinical, from scratch) | 5B & 20B | 277B words (82B UF Health clinical + 195B Pile) | 560 × A100 on HiPerGator, Megatron/NeMo | SOTA biomedical relation extraction (DDI, chemical-disease, drug-target); synthetic text passed clinical Turing tests |
| **Meditron-70B** (medical, continued pretrain of Llama-2) | 7B & 70B | 48.1B-token GAP-Replay (PubMed + 46K clinical guidelines + RedPajama) | 128 × A100 80GB, 16 nodes, Megatron-LLM, BF16 | Beat Llama-2-70B, GPT-3.5, Flan-PaLM on medical reasoning |
| **BioMedLM / PubMedGPT** (biomedical, from scratch) | 2.7B | 300B tokens PubMed (custom 28,896 tokenizer) | 128 × A100 40GB, ~6.25 days | SOTA 50.3% MedQA; beat GPT-Neo 2.7B by ~17 pts; privacy/cost/sustainability benefits |
| **Med-PaLM 2** (medical, fine-tuned PaLM-2) | — | MultiMedQA instruction tuning; ensemble refinement | — | 86.5% MedQA (SOTA at release); physicians preferred its answers on 8/9 clinical axes |
| **Palmyra-Med-70B** (medical, Writer, SFT+DPO) | 70B | Biomedical + custom instruction + DPO | — | 85.9% avg (zero-shot) vs Med-PaLM-2 84%; $10/1M output vs GPT-4 $60 |
| **PharmBERT** (drug labels) | BERT-base | FDA SPL drug labels | — | Beat BERT, ClinicalBERT, BioBERT on drug-label NLP |
| **DeepSeek-V3** (general, MoE — efficiency proof) | 671B total / 37B active | 14.8T tokens | 2.788M H800 GPU-hrs ≈ $5.576M (2,048 H800s) | Matched GPT-4o at ~1/10 the training cost of Llama-3.1-405B |

The message for leadership: **domain data + modest scale + smart post-training repeatedly beats giant general models on niche tasks**, and the most efficient large models (DeepSeek-V3) prove that clever architecture and data curation — not brute spend — drive value.

### 4. Technical Design Choices (Deeper Slides)

**Scaling laws.** Chinchilla (Hoffmann 2022) established ~20 tokens/parameter as compute-optimal (70B model on 1.4T tokens), verified by Cerebras-GPT across 111M–13B (losses stable at 20–40 tokens/param). But for a model we'll run at high inference volume, deliberately "over-training" past Chinchilla is worthwhile: LLaMA-3 8B used ~15T tokens (~1,875 tokens/param, ~10× Chinchilla) to yield a small, inference-cheap model. Sardana et al. (arXiv:2401.00448) formalize this: "as inference demand approaches pre-training data size, the additional cost pushes the optimal tokens-to-parameters ratio towards smaller and longer-trained models." **Recommendation: a 3B–13B model on 300B–1T tokens** (well past Chinchilla-optimal, optimized for inference economics).

**Small-model / data-quality trend.** Phi-1 (1.3B) matched much larger code models using "textbook-quality" data (Gunasekar 2023, arXiv:2306.11644 — trained in 4 days on 8 A100s); Phi-3-mini (3.8B, 3.3T tokens) rivaled far larger models. For pharmacy, curated high-quality proprietary text plus synthetic data can substitute for raw scale.

**Architecture (consensus recipe).** Decoder-only Transformer; pre-norm **RMSNorm**; **RoPE** positional embeddings; **Grouped-Query Attention (GQA)** for a small KV-cache and cheap inference; **SwiGLU** activation (replaced GeLU/ReLU across LLaMA/PaLM "because it works"); **FlashAttention-2/3**; BF16. This is exactly the LLaMA-3 and Meditron-70B stack.

**MoE vs dense.** Mixture-of-Experts (Mixtral 8x7B; DeepSeek-V3's fine-grained DeepSeekMoE plus Multi-head Latent Attention) delivers large capacity at low active-compute cost, but adds training instability, load-balancing complexity, and serving overhead. **Recommendation: start dense at 3–13B (simpler, safer); consider MoE only later**, where experts could specialize by pharmacy subdomain (DDI, claims, prior-auth). DeepSeek-V3's MLA compresses the KV cache ~10× (~200GB→~20GB) — a future option for long-context claims documents.

**Two paths.**
- **Path A — Continued pre-training / DAPT** (Gururangan et al. 2020, "Don't Stop Pretraining," arXiv:2004.10964, ACL 2020): take an open base (Llama/Mistral/Qwen) and continue pre-training on our pharmacy corpus. Cheapest, fastest, proven — this is exactly how Meditron-70B was built. **Recommended first step.**
- **Path B — Full from-scratch** (BloombergGPT / GatorTronGPT / BioMedLM): maximum control over tokenizer, data provenance, and IP; higher cost and risk. **Recommended only after pilot success**, likely at 3–7B where a custom pharmacy tokenizer pays off most.

**Data curation.** De-identification (HIPAA Safe Harbor / Expert Determination) plus NER-based PHI scrubbing; MinHash deduplication; quality filtering; deliberate data mixing (GatorTronGPT's clinical + Pile precedent — roughly 30% clinical / 70% general is one anchor). Synthetic data generation is validated: GatorTronGPT generated 20B synthetic clinical words, and models trained on that synthetic text *outperformed* models trained on real clinical text.

**Post-training pipeline.** SFT on pharmacist-authored instructions (self-instruct / Evol-Instruct) → preference optimization via **DPO** (Direct Preference Optimization — simpler and more stable than RLHF PPO, needs no separate reward model, and is exactly what Palmyra-Med used) → optional RLAIF / constitutional constraints → **RAG** for current formulary and drug facts → guardrails. Cheap per-task adaptation via **LoRA/QLoRA** adapters means one base model can power many pharmacy tasks (DDI, SIG translation, prior-auth, pharmacovigilance) at low marginal cost.

**Evaluation.** MedQA, PubMedQA, MedMCQA, plus pharmacy-specific NAPLEX-style evals and internal benchmarks (à la Bloomberg's internal suite) that reflect real pharmacy workflows.

### 5. Pre-Training vs Post-Training (Executive Framing)
- **Pre-training** builds a base model that learns the "language" of pharmacy from huge unlabeled corpora (expensive, done rarely). This is where our proprietary data becomes the moat.
- **Post-training** (SFT → preference optimization → RL) makes the model follow instructions and behave safely — cheaper, iterative, done on our own compute.
- Modern pipeline: **pretrain → continued/mid-train on domain data → SFT → DPO/preference optimization → RL/RAG/guardrails.** Because post-training and LoRA fine-tuning are cheap, one base model gives us the liberty to spin up many downstream pharmacy applications inexpensively.

### 6. Proposed Architecture (Diagram Content)
End-to-end pipeline for the architecture slide:
1. **Data sources** — e-prescriptions (SIG), pharmacy claims (NCPDP), drug labels (FDA SPL), formulary, RxNorm/NDC/SNOMED terminologies, clinical notes, PubMed/guidelines.
2. **De-identification** — Safe Harbor + Expert Determination, NER PHI scrubbing.
3. **Curation** — MinHash dedup, quality filtering, data mixing, synthetic augmentation.
4. **Domain tokenizer** — custom BPE/SentencePiece for drug names, NDC/RxCUI codes, SIG abbreviations.
5. **Pre-training stack** — decoder-only Transformer layers: RMSNorm → GQA (with RoPE) → SwiGLU FFN (optional MoE FFN later); FlashAttention; BF16; FSDP/Megatron 3D parallelism; gradient checkpointing; cosine/WSD LR schedule.
6. **Checkpointing & eval** — MedQA/PubMedQA/NAPLEX-style + internal pharmacy benchmarks.
7. **Post-training** — SFT → DPO → guardrails.
8. **RAG + guardrails** — live formulary/drug-DB retrieval; safety filters (à la Llama Guard).
9. **Downstream apps (LoRA adapters)** — DDI checking, prior-auth automation, SIG translation, pharmacovigilance/AE detection, formulary Q&A, medication reconciliation.
Plus an **infrastructure view**: GPU cluster (H100/A100), object storage for the corpus, experiment tracking, model registry, and secure-VPC inference.

### 7. Cost Tables (slide-ready)

**Training cost (compute-only, C = 6ND FLOPs, H100 @ 40% MFU on ~990 TFLOP/s BF16, on-demand $2.50–3.00/GPU-hr):**

| Model | Tokens | Compute (FLOPs) | H100 GPU-hours | Cost @ $2.50/hr | Cost @ $3.00/hr |
|---|---|---|---|---|---|
| 3B | 300B | 5.4 × 10²¹ | ~3,788 | ~$9,470 | ~$11,364 |
| 7B | 500B | 2.1 × 10²² | ~14,731 | ~$36,826 | ~$44,192 |
| 13B | 1T | 7.8 × 10²² | ~54,714 | ~$136,785 | ~$164,141 |

*Formula origin: Kaplan et al. 2020 (arXiv:2001.08361) — "C ≈ 6N floating point operations per training token" (forward ≈ 2N, backward ≈ 4N); reused by Chinchilla (arXiv:2203.15556). Add 20–50%+ for failed runs, data prep, checkpointing, eval, storage, and networking. Spot H100 (~$1.50/hr) roughly halves these. MFU is the biggest lever: 50% MFU ≈ ×0.8; 35% MFU ≈ ×1.14.*

**API pricing benchmark (per 1M tokens):**

| Model | Input | Output |
|---|---|---|
| GPT-4o | $2.50 | $10.00 |
| GPT-4o mini | $0.15 | $0.60 |
| Claude 3.5 Sonnet | $3.00 | $15.00 |
| Writer Palmyra-Med (domain model) | — | $10.00 (vs GPT-4 $60) |

At enterprise pharmacy volumes (billions of tokens/month), recurring API costs quickly exceed a one-time training run plus cheap in-house inference — and every one of those API tokens carries PHI-egress risk.

### 8. ROI / Investment Framing for Leadership
- **Build an asset, not a subscription.** Training is CapEx that creates a reusable, appreciating asset (the base model + tokenizer + data pipeline); frontier APIs are perpetual OpEx that scales linearly with usage and never yields ownership.
- **Data as a refinery/moat.** Our proprietary de-identified pharmacy corpus is the crude oil; the model is the refinery. Competitors and frontier labs cannot replicate it because they don't have our data. This is precisely BloombergGPT's thesis (363B proprietary tokens) and GatorTronGPT's (82B words of UF Health clinical text).
- **Honest positioning.** We will *not* out-general GPT-5. We win on four axes only: **our domain, our data, our privacy, our cost.** That is a defensible, winnable strategy.

## Recommendations

**Stage 0 — Foundations (Months 0–2).** Stand up the de-identification pipeline, assemble and curate the pharmacy corpus, build a custom pharmacy tokenizer, and define internal evals (DDI, SIG, prior-auth, formulary Q&A) alongside MedQA/PubMedQA/NAPLEX-style benchmarks. Team: ~4–6 people (ML engineers, a data engineer, 1–2 pharmacist SMEs, compliance).

**Stage 1 — Pilot via continued pre-training + post-training (Months 2–5).** Take an open base (Llama/Mistral/Qwen 3–8B), run DAPT on pharmacy data, then SFT + DPO with pharmacist annotators, and add RAG. Cost: low tens of thousands in compute. **Go/no-go benchmark:** beat GPT-4o on our internal pharmacy eval suite, eliminate PHI egress, and demonstrate lower per-query cost.

**Stage 2 — Scale / from-scratch decision (Months 5–12).** If the pilot clears thresholds, train a 3–7B model (from scratch if tokenizer/IP control justifies it) on 300–500B tokens (~$10–45K compute). Deploy LoRA adapters per task.

**Stage 3 — Production & expansion.** Secure-VPC inference, monitoring, guardrails, and continuous RAG updates for new drug approvals; introduce MoE only if subdomain specialization demands it.

**KPIs:** internal pharmacy-task accuracy vs GPT-4o/Claude; hallucination rate on DDI/dosage; zero PHI egress; cost per 1M tokens vs API; latency; pharmacist-rated safety.

**Thresholds that would change the plan:**
- If Stage-1 DAPT already meets internal targets, **defer from-scratch indefinitely** (cheaper, faster).
- If frontier API prices collapse below in-house inference cost **and** a HIPAA-compliant BAA + deep fine-tuning path emerges, re-evaluate build-vs-buy.
- If long-context claims processing dominates the workload, prioritize MLA/MoE architectures.

## Caveats
- **We cannot and should not try to beat frontier models on general ability.** The honest, winning position is narrow: our domain, our data, our privacy, our cost.
- Cost figures are **compute-only lower bounds.** Real budgets should assume meaningful multipliers for data engineering, failed runs, and MLOps; the $9K–$164K figures are the training-run floor, not the program cost.
- Some cited benchmark and GPU-pricing figures come from vendor/marketing pages and secondary trackers; the core claims (BloombergGPT, GatorTronGPT, Meditron, BioMedLM, Med-PaLM 2, scaling laws, the Sikora DDI study, the Sagar/Amazon Nature Medicine study, NAPLEX) are anchored to arXiv papers and peer-reviewed journals.
- Writer's Palmyra-Med is reportedly being deprecated (mid-2026) in favor of the general Palmyra X5 — cite it as a *historical domain-model benchmark*, not a live product recommendation.
- Medical hallucination is an intrinsic LLM property; an in-house model reduces but does not eliminate it — RAG, guardrails, and human-in-the-loop review remain mandatory for any clinical use.
- De-identification and regulatory sign-off (HIPAA, and FDA where a use case becomes a medical device) are gating dependencies, not afterthoughts.

---
### Source anchors (for the appendix / speaker notes)
- Scaling laws: Kaplan et al. 2020 (arXiv:2001.08361); Hoffmann et al./Chinchilla 2022 (arXiv:2203.15556); inference-aware scaling, Sardana et al. (arXiv:2401.00448); Cerebras-GPT (arXiv:2304.03208).
- Architecture: LLaMA-3 herd (SwiGLU/RMSNorm/RoPE/GQA); DeepSeek-V3 Technical Report (arXiv:2412.19437 — 2.788M H800 GPU-hrs, $5.576M); Phi / "Textbooks Are All You Need" (arXiv:2306.11644, 2309.05463).
- Healthcare precedents: GatorTronGPT (npj Digital Medicine, arXiv:2305.13523); Meditron-70B (arXiv:2311.16079); BioMedLM (arXiv:2403.18421); Med-PaLM 2 (arXiv:2305.09617); PharmBERT (Briefings in Bioinformatics, bbad226); Palmyra-Med (Writer, 2024).
- Pharmacy safety/benchmarks: Sikora et al. GPT-4 DDI (medRxiv doi:10.1101/2024.06.29.24309701); Sagar et al. MEDIC / Amazon Pharmacy (Nature Medicine, doi:10.1038/s41591-024-02933-8); Angel et al. NAPLEX (Am. J. Pharm. Educ. 2024).
- Post-training: DPO (Rafailov et al. 2023); "A Survey on Post-training of LLMs" (arXiv:2503.06072); DAPT (Gururangan et al. 2020, arXiv:2004.10964).
- Terminology/standards: NLM RxNorm technical docs; NCPDP/ONC interoperability filings; FDA SPL.