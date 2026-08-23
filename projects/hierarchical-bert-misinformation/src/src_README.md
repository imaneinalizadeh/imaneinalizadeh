# src/

Source code for the Hierarchical Fake News Detector. Scripts are organised into subfolders by function and numbered in execution order.

---

## Folder Structure

```
src/
├── training/
│   ├── baseline/          ← core hierarchical + flat baseline models
│   ├── ai_generated/      ← AI-Generated Content extension (3 attempts)
│   ├── retraining/        ← V5 improved retraining from error analysis
│   └── extensibility/     ← taxonomy extensibility experiment training scripts
│
├── data_builders/         ← scripts that build datasets (not model training)
│
├── analysis/              ← evaluation and error analysis scripts
│
└── src_README.md
```

---

## Execution Order

| Step | Script | What it does | Output |
|------|--------|-------------|--------|
| 1 | `training/baseline/train_hierarchical.py` | Train the two-stage hierarchical BERT classifier on 564K Fakeddit samples | `baseline/best_model.pt` |
| 2 | `training/baseline/train_flat_baseline.py` | Train flat BERT baseline under identical conditions for fair comparison | `baseline/flat_best_model.pt` |
| 3a | `training/ai_generated/add_ai_attempt1_essays.py` | AI-Generated Attempt 1 — essay dataset, exposes domain mismatch | `ai_generated/hierarchical_v2_best.pt` |
| 3b | `training/ai_generated/add_ai_attempt2_essays.py` | AI-Generated Attempt 2 — Reuters headlines, partial domain fix | `ai_generated/hierarchical_v2_headlines_best.pt` |
| 3c | `training/ai_generated/add_ai_attempt3_essays.py` | AI-Generated Attempt 3 — NYT titles, full domain alignment | `ai_generated/hierarchical_v3_best.pt` |
| 4a | `data_builders/clickbait_dataset_builder.py` | Build Clickbait extension dataset from Fakeddit splits | `data/clickbait/` |
| 4b | *(Clickbait training — inline with step 4a)* | Train Group 1 fine head with Clickbait as new label | `clickbait/hierarchical_v4_checkpoint.pt` |
| 5 | `analysis/generate_error_analysis.py` | Run full error analysis on final 8-category model, save markdown report | `results/error_analysis/Error_Analysis.md` |
| 6 | `training/retraining/train_v5_resume.py` | Retrain with error-analysis-driven fixes — weighted loss, differential LR, label smoothing | `production/hierarchical_v5_best.pt` |
| 7a | `data_builders/round1_conspiracy_dataset.py` | Extensibility Experiment — attempt to build Conspiracy Theory dataset | `data/extensibility/conspiracy_dataset.json` (14 samples — abandoned) |
| 7b | `training/extensibility/round1_conspiracy_train.py` | Extensibility Experiment — train Conspiracy Theory category | Not run — dataset too small |
| 8 | `training/extensibility/round2_propaganda_train.py` | Extensibility Experiment — train Propaganda category from r/propagandaposters | `extensibility/hierarchical_v6_propaganda.pt` |

---

## What This Project Is Trying To Do

This project answers two research questions:

**Question 1:** Does a hierarchical two-stage BERT classifier outperform a flat classifier for fake news detection?

**Question 2:** When does adding a new misinformation category to the hierarchy stop being beneficial?

Scripts 1–2 answer Question 1. Scripts 3a–8 answer Question 2 through iterative taxonomy extension experiments.

---

## Script Details

### `training/baseline/train_hierarchical.py` — Core Hierarchical Model (Step 1)

Trains the main model. The key idea is a **two-stage classification process**:

- Stage 1 (coarse head): assigns each post to one of 3 semantic groups
- Stage 2 (fine head): within that group, predicts the exact category

This is better than a flat classifier because semantically related categories share a fine head and learn from each other, while unrelated categories are separated at the coarse level before fine-grained decisions are made.

**Architecture:**

```
BERT (bert-base-uncased) → 768-dim CLS token
├── Coarse head  → 3 groups (Authentic / Structural Deception / Fabricated)
├── Fine head 0  → 1 class  (True Content)
├── Fine head 1  → 2 classes (Misleading Content, False Connection)
└── Fine head 2  → 3 classes (Satire, Imposter, Manipulated Content)
```

**Loss function:**
```
total_loss = 0.5 × coarse_loss + 0.5 × fine_loss
```

**Hyperparameters:**

| Parameter | Value | Why |
|-----------|-------|-----|
| Base model | bert-base-uncased | Strong NLP baseline, widely validated |
| Learning rate | 2e-5 | Standard for BERT fine-tuning — avoids catastrophic forgetting |
| Batch size | 64 | Maximum that fits T4 GPU memory without OOM |
| Max sequence length | 64 tokens | Fakeddit titles average 8–12 words |
| Epochs | 2 | Loss converged by epoch 2 — training beyond this overfits |
| Optimiser | AdamW | Decoupled weight decay — standard for transformers |
| Warmup | 10% of steps | Prevents large gradient updates at the start |
| Loss alpha | 0.5 | Equal coarse / fine weighting |

**Result:** 82% accuracy · Macro F1 0.77

---

### `training/baseline/train_flat_baseline.py` — Flat Baseline (Step 2)

Trains a flat 6-class BERT classifier for controlled comparison. The only difference is the classification head — a single `nn.Linear(768, 6)` instead of the two-stage hierarchy. Everything else is identical.

**Why we need a baseline:** Without this, we can't know if improvements come from the hierarchical design or just the BERT backbone. The flat baseline isolates the architectural contribution.

**Statistical reliability (seeds 42, 123, 456):**
- Hierarchical: Macro F1 0.77 ± 0.005
- Flat: Macro F1 0.75 ± 0.006
- Paired t-test: p < 0.05

**Result:** 81% accuracy · Macro F1 0.75

---

### `training/ai_generated/add_ai_attempt1_essays.py` — AI-Generated Attempt 1 (Step 3a)

Extends the model with AI-Generated Content (label 6) in Group 2. Dataset: `artem9k/ai-text-detection-pile` — 30,000 samples.

**Selective freezing:** Group 2 fine head reinitialised (3→4 outputs) and retrained. Everything else frozen.

**Why this failed:** Essays average 400+ words; Fakeddit titles average 8–12 words. Domain mismatch — model never predicts AI-Generated at test time.

**Result:** AI-Generated F1 = 0.11

---

### `training/ai_generated/add_ai_attempt2_essays.py` — AI-Generated Attempt 2 (Step 3b)

Dataset: `artnitolog/llm-generated-texts` — Reuters headlines, ~22,000 samples, 15–120 chars. Fixes length mismatch from Attempt 1.

**Why F1 improved but target not reached:** Reuters is formal; Reddit is casual. Style gap remained.

**Result:** AI-Generated F1 = 0.71

---

### `training/ai_generated/add_ai_attempt3_essays.py` — AI-Generated Attempt 3 (Step 3c)

Dataset: `gsingh1-py/train` — NYT titles generated by 5 LLMs (GPT-4o, Llama-8B, Mistral-7B, Qwen-2-72B, Gemma-2-9B). ~25,000 samples, 15–120 chars.

**Why this worked:** NYT titles and Reddit post titles share the same format — short, punchy, single idea. Five LLM sources prevent overfitting to one model's fingerprint.

**Result:** AI-Generated F1 = 0.90 · Overall accuracy 84% · Macro F1 0.81

---

### `data_builders/clickbait_dataset_builder.py` — Clickbait Dataset Builder (Step 4a)

Builds the Clickbait category dataset from existing Fakeddit data — no external source needed. Scores Group 1 posts using subreddit membership and 10 linguistic pattern regexes. Posts scoring above 0.45 are relabelled as Clickbait.

```bash
python src/data_builders/clickbait_dataset_builder.py \
  --input_dir data/fakeddit/ \
  --output_dir data/clickbait/ \
  --verbose
```

**Why Clickbait succeeded in one attempt:** In-domain data by definition. No length or style mismatch.

**Result:** Clickbait F1 = 0.83 · Zero regression · `clickbait/hierarchical_v4_checkpoint.pt`

---

### `analysis/generate_error_analysis.py` — Full Error Analysis (Step 5)

Runs inference on `clickbait/hierarchical_v4_checkpoint.pt` and generates a 15-section markdown report. No training.

**Why this matters:** Tells us exactly where the model fails before we retrain — which categories to upweight, what the routing bottleneck is, where confidence is miscalibrated.

**Key sections:**

| Section | Key finding |
|---------|------------|
| Error breakdown | **71.2% of errors are routing errors** — coarse head is the bottleneck |
| Loss analysis | Imposter Content loss 1.66 — 10× higher than False Connection |
| Calibration | ECE = 0.1385 — model overconfident at every level |
| Hardest examples | 25 cases all require image features — text-only ceiling identified |

**Output:** `results/error_analysis/Error_Analysis.md` · Runtime: ~5 minutes on T4 GPU.

---

### `training/retraining/train_v5_resume.py` — V5 Improved Retraining (Step 6)

Retrains from `clickbait/hierarchical_v4_checkpoint.pt` applying four fixes from the error analysis. Every fix maps directly to a finding.

| Fix | Error analysis finding | Fix applied |
|-----|----------------------|-------------|
| Weighted loss | Imposter loss 1.66, error 45.5% | Imposter 5×, Satire 3×, Manipulated 3×, AI-Gen 2× |
| Coarse head LR = 3e-5 | 71.2% errors are routing errors | Give coarse head highest LR |
| Partial BERT unfreeze | Fabricated/Manip 71.9% coarse confidence | Let encoder adapt to hard cases |
| Label smoothing 0.1 | ECE = 0.1385, 63.3% wrong at >90% confidence | Penalise overconfidence |

**Result (1 epoch):** 82.43% accuracy · Macro F1 0.7755 · Every category improved

---

### `data_builders/round1_conspiracy_dataset.py` — Extensibility Data Search (Step 7a)

Part of Experiment 4: "When does adding a new category stop being beneficial?"

Searches for Conspiracy Theory training data across Fakeddit subreddits, 3 HuggingFace datasets, and 12 linguistic regex patterns.

**Result:** Only 14 usable samples. Fakeddit has no conspiracy subreddits. All HuggingFace datasets failed to load. **Conspiracy Theory abandoned — insufficient data.**

This is itself a finding: some categories cannot be added because Reddit data for them simply does not exist in sufficient quantity.

---

### `training/extensibility/round1_conspiracy_train.py` — Conspiracy Theory Training (Step 7b)

Written but **not run** — 14 samples is not enough to train. Provided for completeness. Would load from `production/hierarchical_v5_best.pt` and add Conspiracy Theory as label 8 in Group 2.

---

### `training/extensibility/round2_propaganda_train.py` — Propaganda Training (Step 8)

Adds Propaganda (label 8) to Group 2. Uses Fakeddit `r/propagandaposters` — 13,456 train / 1,455 test samples.

**Viability criteria (fail ANY one = stop):**

| # | Criterion | Threshold | Why |
|---|-----------|-----------|-----|
| 1 | New category F1 | > 0.65 | Below this the model is not reliably identifying the category |
| 2 | Max existing F1 drop | < 0.03 | More than 3 points means adding this category hurts the model |
| 3 | Routing error rate | < +5% | Coarse head destabilisation means the whole architecture is struggling |
| 4 | ECE | < 0.20 | Overconfidence makes the model unreliable in deployment |

**Results:**

| Epoch | Propaganda F1 | Routing | ECE | Verdict |
|-------|--------------|---------|-----|---------|
| 1 | 0.1333 | 73.9% | 0.048 | ✗ FAIL (F1, Routing) |
| 2 | 0.1335 | 73.4% | 0.053 | ✗ FAIL (F1, Routing) |
| 3 | 0.1339 | 73.9% | 0.055 | ✗ FAIL (F1, Routing) |

**Why it failed:** Propaganda captions (*"join the army"*, *"dreams will come true"*) are linguistically indistinguishable from True Content — the deception is in the visual imagery, not the text. Routing also broke down, making this a more severe failure than Junk News.

Full analysis: `results/experiment4/extensibility_analysis.md`

---

## Taxonomy

```
Group 0 — Authentic
└── True Content (label 0)

Group 1 — Structural Deception
├── Misleading Content (label 2)    — real event, wrong context
├── False Connection (label 4)      — real image, wrong headline
└── Clickbait (label 7)             ← added step 4

Group 2 — Fabricated/Manipulated
├── Satire/Parody (label 1)         — comedy presented as news
├── Imposter Content (label 3)      — fake source impersonating real one
├── Manipulated Content (label 5)   — doctored image or video
└── AI-Generated Content (label 6)  ← added steps 3a–3c
```

Extension attempts that failed viability:
- Junk News → Group 1 label 8 — failed: intra-group overlap with Misleading/Clickbait
- Propaganda → Group 2 label 8 — failed: text-thin signal, routing breakdown

---

## Label Maps

```python
# Final 8-category maps (steps 4–6)
FINE_TO_COARSE = {0:0, 1:2, 2:1, 3:2, 4:1, 5:2, 6:2, 7:1}

LOCAL_TO_FINE = {
    (0, 0): 0,                                    # Group 0 → True Content
    (1, 0): 2, (1, 1): 4, (1, 2): 7,            # Group 1 → Misleading, False Connection, Clickbait
    (2, 0): 1, (2, 1): 3, (2, 2): 5, (2, 3): 6  # Group 2 → Satire, Imposter, Manipulated, AI-Gen
}
```

---

## Model Checkpoints

| File | Location | Step | Accuracy | Macro F1 |
|------|----------|------|----------|----------|
| `best_model.pt` | `models/baseline/` | 1 | 82% | 0.77 |
| `flat_best_model.pt` | `models/baseline/` | 2 | 81% | 0.75 |
| `hierarchical_v2_best.pt` | `models/ai_generated/` | 3a | — | 0.11 (AI-Gen) |
| `hierarchical_v2_headlines_best.pt` | `models/ai_generated/` | 3b | — | 0.71 (AI-Gen) |
| `hierarchical_v3_best.pt` | `models/ai_generated/` | 3c | 84% | 0.81 |
| `hierarchical_v4_best.pt` | `models/clickbait/` | 3c+ | 84% | 0.81 |
| `hierarchical_v4_checkpoint.pt` | `models/clickbait/` | 4 | 80.89% | 0.7665 |
| `hierarchical_v5_best.pt` | `models/production/` | 6 | **82.43%** | **0.7755** |
| `hierarchical_v6_junknews.pt` | `models/extensibility/` | Exp4 | — | — |
| `hierarchical_v6_propaganda.pt` | `models/extensibility/` | 8 | — | — |

---

## Infrastructure

| Resource | Role |
|----------|------|
| Google Colab T4 GPU | Primary training environment (~6 hours per full run) |
| EIDF eidf018 cluster | Backup GPU (eidf-vdi.epcc.ed.ac.uk) |
| Google Drive (ieinalizadeh) | Model checkpoints |
| Google Drive (punchwhit3) | Fakeddit dataset TSV files |

---

## Reproduction

```bash
# Step 1-2: Baseline
python src/training/baseline/train_hierarchical.py
python src/training/baseline/train_flat_baseline.py

# Step 3: AI-Generated Content extension
python src/training/ai_generated/add_ai_attempt1_essays.py
python src/training/ai_generated/add_ai_attempt2_essays.py
python src/training/ai_generated/add_ai_attempt3_essays.py

# Step 4: Clickbait extension
python src/data_builders/clickbait_dataset_builder.py --input_dir data/fakeddit/ --output_dir data/clickbait/ --verbose

# Step 5: Error analysis
python src/analysis/generate_error_analysis.py

# Step 6: V5 retraining
python src/training/retraining/train_v5_resume.py

# Extensibility experiment
python src/data_builders/round1_conspiracy_dataset.py   # finds ~14 samples — abandoned
python src/training/extensibility/round2_propaganda_train.py
```