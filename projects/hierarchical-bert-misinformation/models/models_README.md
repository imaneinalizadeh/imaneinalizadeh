# models/

This folder contains all trained model checkpoints for the Hierarchical Multi-Class Fake News Detection project. All files are stored via Git LFS. Each checkpoint is the best-epoch output of its training run and serves as the base for the next extension step.

---

## Checkpoint Index

| File | Step | Categories | Accuracy | Macro F1 | Key Metric |
|------|------|-----------|----------|----------|------------|
| `baseline/best_model.pt` | 1 | 6 | 82% | 0.77 | Baseline hierarchical |
| `baseline/flat_best_model.pt` | 2 | 6 | 81% | 0.75 | Flat baseline comparison |
| `ai_generated/hierarchical_v2_best.pt` | 3a | 7 | — | — | AI-Gen F1 0.11 (domain mismatch) |
| `ai_generated/hierarchical_v2_headlines_best.pt` | 3b | 7 | — | — | AI-Gen F1 0.71 (Reuters fix) |
| `ai_generated/hierarchical_v3_best.pt` | 3c | 7 | 84% | 0.81 | AI-Gen F1 0.90 (target achieved) |
| `clickbait/hierarchical_v4_best.pt` | 3c+ | 7 | 84% | 0.81 | Retrained with weighted loss + differential LR |
| `clickbait/hierarchical_v4_checkpoint.pt` | 4 | 8 | 80.89% | 0.7665 | Clickbait F1 0.83 — final 8-category model |
| `production/hierarchical_v5_best.pt` | 5 | 8 | 82.43% | 0.7755 | Error analysis fixes applied |
| `extensibility/hierarchical_v6_junknews.pt` | Exp4 | 9 (attempt) | — | — | Junk News F1 0.155 — **not viable** |
| `extensibility/hierarchical_v6_propaganda.pt` | Exp4 | 9 (attempt) | — | — | Propaganda F1 0.133 — **not viable** |

---

## Dependency Chain

```
best_model.pt  (step 1 — 6 categories)
├── flat_best_model.pt                     (step 2 — trained independently, same base)
└── hierarchical_v2_best.pt                (step 3a — Group 2 head extended to 4 outputs)
    └── hierarchical_v2_headlines_best.pt  (step 3b — Group 2 retrained, Reuters data)
        └── hierarchical_v3_best.pt        (step 3c — Group 2 retrained, NYT data)
            └── hierarchical_v4_best.pt    (step 3c+ — full retrain, weighted loss)
                └── hierarchical_v4_checkpoint.pt  (step 4 — Group 1 extended, Clickbait)
                    └── hierarchical_v5_best.pt    (step 5 — error analysis fixes)
                        ├── hierarchical_v6_junknews.pt   (Exp4 Round 1 — failed)
                        └── hierarchical_v6_propaganda.pt (Exp4 Round 2 — failed)
```

Each step loads the previous checkpoint and retrains only the relevant components. The BERT encoder is frozen (or partially unfrozen at lr=1e-5 from step 3c onwards) to prevent catastrophic forgetting. The two v6 checkpoints both failed viability criteria and are retained for reference only — `production/hierarchical_v5_best.pt` remains the recommended production model.

---

## Architecture

All checkpoints share the same base architecture. The only differences are the number of outputs in the fine heads as the taxonomy is extended.

```python
class HierarchicalFakeNewsClassifier(nn.Module):
    def __init__(self, g1_classes=2, g2_classes=3):
        super().__init__()
        self.bert        = BertModel.from_pretrained("bert-base-uncased")  # 768-dim CLS
        h                = self.bert.config.hidden_size
        self.coarse_head = nn.Linear(h, 3)           # always 3 groups
        self.fine_heads  = nn.ModuleList([
            nn.Linear(h, 1),           # Group 0: True Content (always 1)
            nn.Linear(h, g1_classes),  # Group 1: 2→3→4 as taxonomy grows
            nn.Linear(h, g2_classes),  # Group 2: 3→4→5 as taxonomy grows
        ])
```

| Checkpoint | G1 outputs | G2 outputs | Total categories |
|-----------|-----------|-----------|-----------------|
| `baseline/best_model.pt` | 2 | 3 | 6 |
| `baseline/flat_best_model.pt` | N/A | N/A | 6 (flat head) |
| `ai_generated/hierarchical_v2_best.pt` | 2 | 4 | 7 |
| `ai_generated/hierarchical_v2_headlines_best.pt` | 2 | 4 | 7 |
| `ai_generated/hierarchical_v3_best.pt` | 2 | 4 | 7 |
| `clickbait/hierarchical_v4_best.pt` | 2 | 4 | 7 |
| `clickbait/hierarchical_v4_checkpoint.pt` | 3 | 4 | 8 |
| `production/hierarchical_v5_best.pt` | 3 | 4 | 8 |
| `extensibility/hierarchical_v6_junknews.pt` | **4** | 4 | 9 (attempt) |
| `extensibility/hierarchical_v6_propaganda.pt` | 3 | **5** | 9 (attempt) |

---

## Label Maps

```python
# 8-category final maps (steps 4–5) — recommended production maps
FINE_TO_COARSE = {0:0, 1:2, 2:1, 3:2, 4:1, 5:2, 6:2, 7:1}

LOCAL_TO_FINE = {
    (0, 0): 0,                                    # Group 0 → True Content
    (1, 0): 2, (1, 1): 4, (1, 2): 7,            # Group 1 → Misleading, False Connection, Clickbait
    (2, 0): 1, (2, 1): 3, (2, 2): 5, (2, 3): 6  # Group 2 → Satire, Imposter, Manipulated, AI-Gen
}

# 9-category maps (Exp4 — Junk News attempt, Group 1 expanded)
LOCAL_TO_FINE_JUNK = {
    (0, 0): 0,
    (1, 0): 2, (1, 1): 4, (1, 2): 7, (1, 3): 8,  # Group 1 adds Junk News
    (2, 0): 1, (2, 1): 3, (2, 2): 5, (2, 3): 6
}

# 9-category maps (Exp4 — Propaganda attempt, Group 2 expanded)
LOCAL_TO_FINE_PROP = {
    (0, 0): 0,
    (1, 0): 2, (1, 1): 4, (1, 2): 7,
    (2, 0): 1, (2, 1): 3, (2, 2): 5, (2, 3): 6, (2, 4): 8  # Group 2 adds Propaganda
}
```

---

## Checkpoint Details

### `baseline/best_model.pt` — Hierarchical Baseline (Step 1)

Core hierarchical model trained on full 564,000 sample Fakeddit training set. Starting point for all taxonomy extension experiments.

**Training:** 2 epochs · batch size 64 · lr=2e-5 · AdamW · loss alpha=0.5

**Per-class results (59,319 test samples):**

| Category | Precision | Recall | F1 | Support |
|----------|-----------|--------|-----|---------|
| True Content | 0.87 | 0.87 | 0.87 | 23,507 |
| Satire/Parody | 0.78 | 0.70 | 0.73 | 3,514 |
| Misleading Content | 0.73 | 0.72 | 0.73 | 11,297 |
| Imposter Content | 0.83 | 0.49 | 0.62 | 1,224 |
| False Connection | 0.82 | 0.88 | 0.85 | 17,472 |
| Manipulated Content | 0.83 | 0.82 | 0.83 | 2,305 |
| **Macro avg** | **0.81** | **0.75** | **0.77** | 59,319 |

**Overall accuracy: 82%**

---

### `baseline/flat_best_model.pt` — Flat Baseline (Step 2)

Flat 6-class BERT classifier trained under identical conditions. The only difference from `baseline/best_model.pt` is the classification head — a single `nn.Linear(768, 6)` with no group structure.

**Per-class F1 vs hierarchical:**

| Category | Flat F1 | Hier F1 | Delta |
|----------|---------|---------|-------|
| True Content | 0.86 | 0.87 | +0.01 |
| Satire/Parody | 0.69 | 0.73 | **+0.04** |
| Misleading Content | 0.72 | 0.73 | +0.01 |
| Imposter Content | 0.58 | 0.62 | **+0.04** |
| False Connection | 0.85 | 0.85 | 0.00 |
| Manipulated Content | 0.80 | 0.83 | +0.03 |
| **Macro F1** | **0.75** | **0.77** | **+0.02** |

**Statistical reliability (seeds 42, 123, 456):**
- Flat: Macro F1 0.75 ± 0.006
- Hierarchical: Macro F1 0.77 ± 0.005
- Paired t-test: **p < 0.05** — improvement is statistically significant

---

### `ai_generated/hierarchical_v2_best.pt` — AI-Generated Attempt 1 (Step 3a)

Extends `baseline/best_model.pt` with AI-Generated Content (label 6) in Group 2. Dataset: `artem9k/ai-text-detection-pile` — 30,000 samples from GPT-2, GPT-3, ChatGPT, GPT-J.

**What changed:** Group 2 fine head reinitialised from 3 → 4 outputs and retrained. BERT, coarse head, Group 0 and Group 1 fine heads frozen.

| Metric | Value | Interpretation |
|--------|-------|---------------|
| F1 | 0.11 | Near-zero — domain mismatch |
| Precision | 0.94 | When predicted, almost always correct |
| Recall | 0.18 | Model almost never predicts AI-Generated |

**Why:** Essays average 400+ words; Fakeddit titles average 8–12 words. Domain mismatch — deliberate documented finding.

---

### `ai_generated/hierarchical_v2_headlines_best.pt` — AI-Generated Attempt 2 (Step 3b)

Reuters-style headlines (`artnitolog/llm-generated-texts`) — ~22,000 samples, 15–120 chars. Addresses length mismatch from Attempt 1.

| Metric | Value | Change vs Attempt 1 |
|--------|-------|-------------------|
| F1 | 0.71 | **+0.60** |
| Precision | 0.78 | −0.16 |
| Recall | 0.61 | **+0.43** |

---

### `ai_generated/hierarchical_v3_best.pt` — AI-Generated Attempt 3 (Step 3c)

NYT titles generated by 5 LLMs (`gsingh1-py/train`) — ~25,000 samples, 15–120 chars.

| Metric | Value | Change vs Attempt 2 |
|--------|-------|-------------------|
| F1 | **0.90** | +0.19 |
| Precision | 0.92 | +0.14 |
| Recall | 0.86 | +0.25 |

**Overall: 84% accuracy · Macro F1 0.81**

---

### `clickbait/hierarchical_v4_best.pt` — Improved V4 (Step 3c+)

Full retrain with improved setup: weighted loss (AI-Gen 2×, Imposter 3×, Satire 2×), differential LRs (BERT 1e-5, heads 2e-5), partial BERT unfreeze.

**Result: 84% accuracy · Macro F1 0.81 · AI-Gen F1 0.90**

---

### `clickbait/hierarchical_v4_checkpoint.pt` — Clickbait Extension (Step 4)

Extends `clickbait/hierarchical_v4_best.pt` with Clickbait (label 7) in Group 1. Group 1 fine head expanded 2 → 3 outputs.

| Category | Precision | Recall | F1 |
|----------|-----------|--------|-----|
| True Content | 0.8886 | 0.8320 | 0.8594 |
| Satire/Parody | 0.7726 | 0.6855 | 0.7265 |
| Misleading Content | 0.6809 | 0.7637 | 0.7199 |
| Imposter Content | 0.7520 | 0.5449 | 0.6319 |
| False Connection | 0.8356 | 0.8507 | 0.8431 |
| Manipulated Content | 0.8303 | 0.8065 | 0.8182 |
| **Macro avg** | **0.7933** | **0.7472** | **0.7665** |

**Overall: 80.89% accuracy · Weighted F1 0.8139 · Clickbait F1 0.83**
Zero regression: Adding Clickbait changed only 16 predictions out of 59,319.

---

### `production/hierarchical_v5_best.pt` — Error Analysis Fixes (Step 5)

Retrains from `clickbait/hierarchical_v4_checkpoint.pt` applying four fixes from error analysis.

| Fix | Detail | Problem targeted |
|-----|--------|-----------------|
| Weighted loss | Imposter 5×, Satire 3×, Manipulated 3×, AI-Gen 2× | Imposter loss 1.66, error 45.5% |
| Coarse head LR = 3e-5 | Higher than fine heads | 71.2% routing error bottleneck |
| Partial BERT unfreeze lr=1e-5 | Encoder adapts to hard cases | Fabricated/Manip 71.9% coarse confidence |
| Label smoothing 0.1 | Penalises overconfidence | ECE = 0.1385 |

| Category | v4 F1 | v5 F1 | Delta |
|----------|-------|-------|-------|
| True Content | 0.8594 | 0.868 | +0.009 |
| Satire/Parody | 0.7265 | 0.730 | +0.004 |
| Misleading Content | 0.7199 | 0.733 | +0.013 |
| Imposter Content | 0.6319 | 0.640 | +0.008 |
| False Connection | 0.8431 | 0.850 | +0.007 |
| Manipulated Content | 0.8182 | 0.831 | +0.013 |

**Overall: 82.43% accuracy · Macro F1 0.7755** ← **recommended production model**

---

### `extensibility/hierarchical_v6_junknews.pt` — Junk News Attempt (Experiment 4, Round 1)

**⚠ Not recommended for production use — failed viability criteria.**

Extends `production/hierarchical_v5_best.pt` with Junk News (label 8) in Group 1. Group 1 fine head expanded 3 → 4 outputs. Dataset: `GonzaloA/fake_news` label=0 — 10,736 politically charged partisan headlines.

**Viability check results (epoch 2):**

| Criterion | Threshold | Result | Verdict |
|-----------|-----------|--------|---------|
| New category F1 > 0.65 | > 0.65 | **0.1546** | ✗ FAIL |
| Max existing drop < 0.03 | < 0.03 | 0.000 | ✓ |
| Routing rate < +5% | < 81.2% | 74.5% | ✓ |
| ECE < 0.20 | < 0.20 | 0.046 | ✓ |

**Why it failed:** Junk News shares surface features with Misleading Content and Clickbait — politically charged partisan framing overlaps with both neighbours in the Group 1 embedding space. The model cannot find a separating boundary despite adequate training data. This is a **representational overlap failure**, not a data quantity failure.

---

### `extensibility/hierarchical_v6_propaganda.pt` — Propaganda Attempt (Experiment 4, Round 2)

**⚠ Not recommended for production use — failed viability criteria.**

Extends `production/hierarchical_v5_best.pt` with Propaganda (label 8) in Group 2. Group 2 fine head expanded 4 → 5 outputs. Dataset: Fakeddit `r/propagandaposters` — 13,456 train / 1,455 test samples (previously folded into Manipulated Content in the original taxonomy).

**Viability check results (epoch 1, best checkpoint):**

| Criterion | Threshold | Result | Verdict |
|-----------|-----------|--------|---------|
| New category F1 > 0.65 | > 0.65 | **0.1333** | ✗ FAIL |
| Max existing drop < 0.03 | < 0.03 | -0.022 | ✓ |
| Routing rate < +5% | < 71.3% | **73.9%** | ✗ FAIL |
| ECE < 0.20 | < 0.20 | 0.048 | ✓ |

**Why it failed:** Propaganda posters are fundamentally visual artifacts — the persuasive content is carried by imagery (art, symbolism, composition), not the text caption. Captions like *"join the army"* or *"dreams will come true"* are linguistically indistinguishable from True Content. This caused both discriminability failure (F1 0.13) AND routing failure — the coarse head could not reliably identify these as Group 2 content from text alone. This is a **text-only modality limitation**.

**This is a more severe failure than Junk News** — routing broke down in addition to fine-grained discrimination, indicating the signal is too thin even for group-level classification.

See `results/experiment4_extensibility_analysis.md` for the full cross-attempt analysis and formal answer to the extensibility research question.

---

## Loading Checkpoints

```python
import torch
import torch.nn as nn
from transformers import BertModel

class HierarchicalFakeNewsClassifier(nn.Module):
    def __init__(self, g1_classes=2, g2_classes=3):
        super().__init__()
        self.bert        = BertModel.from_pretrained("bert-base-uncased")
        h                = self.bert.config.hidden_size
        self.coarse_head = nn.Linear(h, 3)
        self.fine_heads  = nn.ModuleList([
            nn.Linear(h, 1),
            nn.Linear(h, g1_classes),
            nn.Linear(h, g2_classes),
        ])

    def forward(self, input_ids, attention_mask):
        cls = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        return self.coarse_head(cls), [h(cls) for h in self.fine_heads]

# Load 6-category model (steps 1–2)
model = HierarchicalFakeNewsClassifier(g1_classes=2, g2_classes=3)
model.load_state_dict(torch.load("models/best_model.pt", map_location="cpu"))

# Load 7-category model (steps 3a–3c+)
model = HierarchicalFakeNewsClassifier(g1_classes=2, g2_classes=4)
model.load_state_dict(torch.load("models/hierarchical_v4_best.pt", map_location="cpu"))

# Load 8-category model (steps 4–5) — RECOMMENDED
model = HierarchicalFakeNewsClassifier(g1_classes=3, g2_classes=4)
model.load_state_dict(torch.load("models/hierarchical_v5_best.pt", map_location="cpu"))

# Load Junk News attempt (step Exp4 Round 1 — not recommended)
model = HierarchicalFakeNewsClassifier(g1_classes=4, g2_classes=4)
model.load_state_dict(torch.load("models/hierarchical_v6_junknews.pt", map_location="cpu"))

# Load Propaganda attempt (step Exp4 Round 2 — not recommended)
model = HierarchicalFakeNewsClassifier(g1_classes=3, g2_classes=5)
model.load_state_dict(torch.load("models/hierarchical_v6_propaganda.pt", map_location="cpu"))

model.eval()
```

**Partial loading for taxonomy extension:**

```python
base_state = torch.load("models/hierarchical_v5_best.pt", map_location="cpu")
new_state  = model.state_dict()

for k, v in base_state.items():
    if k in new_state and new_state[k].shape == v.shape:
        new_state[k] = v  # copy matching weights
    # mismatched shapes (extended fine head) left randomly initialised

model.load_state_dict(new_state)
```

---

## Storage

All checkpoints stored via **Git LFS** in this repository and backed up to Google Drive:

| Drive | Files |
|-------|-------|
| ieinalizadeh Drive (root) | `clickbait/hierarchical_v4_best.pt`, `clickbait/hierarchical_v4_checkpoint.pt`, `production/hierarchical_v5_best.pt`, `extensibility/hierarchical_v6_junknews.pt`, `extensibility/hierarchical_v6_propaganda.pt` |
| punchwhit3 Drive (`/content/drive/MyDrive/fakeddit/`) | `baseline/best_model.pt`, `baseline/flat_best_model.pt`, `ai_generated/hierarchical_v2_best.pt`, `ai_generated/hierarchical_v2_headlines_best.pt`, `ai_generated/hierarchical_v3_best.pt` |

Checkpoint size: ~438MB per file · Total: ~4.4GB across all 10 checkpoints.

---

*Author: Iman Ein Alizadeh (s2901349) · University of Edinburgh EPCC · MSc Dissertation 2025–26*
*Supervisor: Oliver Brown*