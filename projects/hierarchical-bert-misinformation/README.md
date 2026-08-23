# Hierarchical Multi-Class Fake News Detection with Extensible Taxonomy

**Student:** Iman Ein Alizadeh (S2901349)
**Programme:** MSc Computer Science (EPCC), University of Edinburgh
**Academic Year:** 2025–26
**Supervisor:** Mr Oliver Brown


---

## What This Project Does

Fake news is not one thing — it takes many forms. A satirical article, a manipulated image, an AI-generated headline, and a misleading caption all deceive in fundamentally different ways. Most detection systems collapse all of these into a single binary decision: real or fake. This project argues that the *type* of deception matters, and that a classifier aware of the semantic relationships between misinformation categories will outperform one that ignores them.

This project builds a **hierarchical BERT classifier** for fine-grained fake news detection on the Fakeddit dataset — 564,000 Reddit posts across 6 misinformation categories. The classifier uses a two-stage architecture: a coarse head first routes each post to a semantic group, then a fine head identifies the exact category within that group and the design allows new categories to be added without retraining the whole model.

Three research questions are answered in sequence:

---

## Research Question 1 — Is Hierarchical Better Than Flat?

A hierarchical classifier groups semantically related categories together before making fine-grained decisions. A flat classifier treats all categories as equally unrelated and decides between them in a single step.

**The experiment:** Train both under identical conditions — same backbone (BERT), same data, same hyperparameters, same random seeds. Only the classification head differs.

**Result:**

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Flat BERT baseline | 81% | 0.75 |
| Hierarchical BERT (ours) | 82% | **0.77** |

The improvement is small but statistically significant (p < 0.05, validated across 3 seeds). The hierarchy benefits minority classes most — Satire/Parody +0.04 F1, Imposter Content +0.04 F1 — because grouping semantically related categories gives the fine head a stronger prior before making its decision.

**How the architecture works:**

```
Input title → BERT encoder → 768-dim vector
                                    │
                            ┌───────▼───────┐
                            │  Coarse Head  │  ← Stage 1: which GROUP?
                            └───────┬───────┘
               ┌───────────────────┼───────────────────┐
               ▼                   ▼                   ▼
         Group 0               Group 1             Group 2
      [Authentic]        [Struct. Deception]    [Fabricated]
           │                     │                   │
      Fine Head 0           Fine Head 1         Fine Head 2
           │                     │                   │
     True Content        Misleading Content     Satire/Parody
                         False Connection       Imposter Content
                         Clickbait              Manipulated Content
                                                AI-Generated Content
```

Stage 1 (coarse head) routes the post to the right group. Stage 2 (fine head) makes the exact category decision within that group. The two stages share the same BERT encoder but have independent classification heads.

**The key insight:** The flat model must learn to separate True Content from Satire using the same head that also separates Satire from Manipulated Content. The hierarchical model separates these at the group level first, then handles within-group distinctions independently — a much simpler per-head decision.

**How freezing saves time when adding new categories:**

This is where the hierarchical design becomes practically valuable. Adding a new category does not require retraining the entire model from scratch.

```
Adding AI-Generated Content to Group 2:

BERT encoder        → FROZEN   ✓  (110M params — unchanged, ~6hrs saved)
Coarse head         → FROZEN   ✓  (routing unchanged)
Fine head 0         → FROZEN   ✓  (True Content unaffected)
Fine head 1         → FROZEN   ✓  (Group 1 categories unaffected)
Fine head 2         → RETRAIN  ←  (only this head learns AI-Generated)
```

Only the fine head for the affected group is retrained — a tiny fraction of the total parameters. The BERT encoder (110M parameters, ~6 hours to train) stays completely frozen, so each new category adds only ~30 minutes of retraining rather than 6+ hours from scratch. This is the core extensibility advantage of the hierarchical design.

Full results: `results/experiment1/hierarchical_vs_flat.md`

---

## Research Question 2 — Can We Add New Categories?

The original Fakeddit taxonomy has 6 categories. Two new ones were added iteratively using the selective freezing strategy described above.

**AI-Generated Content** — added across 3 iterative attempts:

| Attempt | Dataset | Avg length | F1 |
|---------|---------|-----------|-----|
| 1 | GPT essay dataset | 400+ words | 0.11 |
| 2 | Reuters headlines | 80–120 chars | 0.71 |
| 3 | NYT titles (5 LLMs) | 40–80 chars | **0.90** |

The key finding: domain match between training data and target distribution is critical. Reddit titles are short and punchy. Only when training data matched that format did the model learn to detect AI-Generated content reliably.

**Clickbait** — added in a single attempt using Fakeddit's own data:

| Metric | Value |
|--------|-------|
| Clickbait F1 | 0.83 |
| Regression on existing categories | Zero |
| Predictions changed out of 59,319 | 16 |

Clickbait succeeded immediately because the data was already in-domain. No length mismatch, no style mismatch — lesson directly applied from the AI-Generated Content experiments.

**Final 8-category model accuracy: 82.43% · Macro F1 0.7755**

Full results: `results/experiment2/` and `results/experiment3/`

---

## Research Question 3 — When Does Adding a Category Stop Working?

Having successfully extended the taxonomy twice, the question becomes: where is the limit? This experiment adds new categories systematically and measures each against four strict criteria. All four must pass — failing one is enough to stop.

**The four criteria:**

| # | Criterion | Threshold | What it catches |
|---|-----------|-----------|----------------|
| 1 | New category F1 | > 0.65 | Category not learned — adding it makes the system less trustworthy |
| 2 | Max existing F1 drop | < 0.03 | Hidden regression — adding the category damages existing performance |
| 3 | Routing error rate increase | < +5% | Architecture destabilisation — coarse head confused by new category |
| 4 | ECE (calibration error) | < 0.20 | Confidence scores become unreliable |

**Four attempts, two trained:**

| Attempt | Category | Samples | Result |
|---------|----------|---------|--------|
| 1 | Conspiracy Theory | 14 | Abandoned — no training data exists |
| 2 | Rumour | 1,419 (noisy) | Abandoned — insufficient clean data |
| 3 | Junk News | 10,736 | **Trained — FAILED Criterion 1** (F1 0.155) |
| 4 | Propaganda | 13,456 | **Trained — FAILED Criteria 1 & 3** (F1 0.133, routing +7.6%) |

**The answer:** The 8-category taxonomy is at or near the practical extensibility limit for text-only hierarchical BERT on this data. Both trained attempts failed because the new categories' linguistic fingerprints were not distinct enough from existing neighbours in the BERT embedding space. Propaganda additionally failed routing — because propaganda poster captions are visually deceptive, not textually. The text alone carries no distinguishing signal.

Criteria 2 and 4 passed every time — the model's existing knowledge and calibration are robust to extension attempts. The binding constraint is always Criterion 1.

**Future work:** Both failures point to the same solution — multimodal extension. Adding a visual encoder (ResNet or ViT) alongside BERT would provide the image features needed to resolve image-dependent confusions. Junk News's partisan framing and Propaganda's visual symbolism both require seeing the image, not just reading the caption.

Full results: `results/experiment4/extensibility_analysis.md`

---

## Taxonomy

```
Group 0 — Authentic
└── True Content (label 0)

Group 1 — Structural Deception
├── Misleading Content (label 2)    — real event, wrong context
├── False Connection (label 4)      — real image, wrong headline
└── Clickbait (label 7)             ← added successfully

Group 2 — Fabricated/Manipulated
├── Satire/Parody (label 1)         — comedy presented as news
├── Imposter Content (label 3)      — fake source impersonating real one
├── Manipulated Content (label 5)   — doctored image or video
└── AI-Generated Content (label 6)  ← added successfully
```

Failed extensions: Junk News (Group 1, intra-group overlap), Propaganda (Group 2, text-thin signal)

---

## Repository Structure

```
s2901349/
│
├── data/
│   ├── fakeddit/          ← train.tsv, validate.tsv, test_public.tsv
│   ├── ai_generated/      ← AI-Generated Content experiment data
│   ├── clickbait/         ← Clickbait extension dataset
│   └── extensibility/     ← conspiracy, junknews, propaganda datasets
│
├── src/
│   ├── training/
│   │   ├── baseline/      ← train_hierarchical.py, train_flat_baseline.py
│   │   ├── ai_generated/  ← add_ai_attempt1/2/3_essays.py
│   │   ├── retraining/    ← train_v5_resume.py
│   │   └── extensibility/ ← round1_conspiracy_train.py, round2_propaganda_train.py
│   ├── data_builders/     ← clickbait_dataset_builder.py, round1_conspiracy_dataset.py
│   └── analysis/          ← generate_error_analysis.py
│
├── models/
│   ├── baseline/          ← best_model.pt, flat_best_model.pt
│   ├── ai_generated/      ← hierarchical_v2/v3_best.pt
│   ├── clickbait/         ← hierarchical_v4_best.pt, hierarchical_v4_checkpoint.pt
│   ├── production/        ← hierarchical_v5_best.pt ← recommended
│   └── extensibility/     ← hierarchical_v6_junknews.pt, hierarchical_v6_propaganda.pt
│
├── results/
│   ├── experiment1/       ← hierarchical vs flat
│   ├── experiment2/       ← AI-Generated Content (3 attempts)
│   ├── experiment3/       ← Clickbait extension
│   ├── experiment4/       ← taxonomy extensibility analysis
│   └── error_analysis/    ← full post-Clickbait error analysis
│
└── submission/
    ├── S2901349-PP-FeasibilityReport.pdf
    └── s2901349_presentation.pdf
```

---

## Model Checkpoints

All stored via Git LFS. Recommended model: `models/production/hierarchical_v5_best.pt`

| File | Location | Accuracy | Macro F1 |
|------|----------|----------|----------|
| `best_model.pt` | `models/baseline/` | 82% | 0.77 |
| `flat_best_model.pt` | `models/baseline/` | 81% | 0.75 |
| `hierarchical_v2_best.pt` | `models/ai_generated/` | — | 0.11 (AI-Gen) |
| `hierarchical_v2_headlines_best.pt` | `models/ai_generated/` | — | 0.71 (AI-Gen) |
| `hierarchical_v3_best.pt` | `models/ai_generated/` | — | 0.90 (AI-Gen) |
| `hierarchical_v4_best.pt` | `models/clickbait/` | 84% | 0.81 |
| `hierarchical_v4_checkpoint.pt` | `models/clickbait/` | 80.89% | 0.7665 |
| `hierarchical_v5_best.pt` | `models/production/` ← **recommended** | **82.43%** | **0.7755** |
| `hierarchical_v6_junknews.pt` | `models/extensibility/` | — | — |
| `hierarchical_v6_propaganda.pt` | `models/extensibility/` | — | — |

---

## Completed Work

- ✅ **RQ1** — Hierarchical vs flat: Macro F1 0.77 vs 0.75, p < 0.05
- ✅ **RQ2** — AI-Generated Content: 3 attempts, F1 0.11 → 0.90
- ✅ **RQ2** — Clickbait: F1 0.83, zero regression
- ✅ **RQ3** — Extensibility: 4 attempts, formal criteria, limit identified
- ✅ Error analysis: 71.2% routing errors, ECE 0.1385 — drove V5 retraining
- ✅ V5 retraining: 82.43% accuracy, every category improved
- ✅ Report submitted 17 April 2026
- ✅ Presentation delivered 20 May 2026

---

## References

Nakamura, K., Levy, S., & Wang, W. Y. (2020). r/Fakeddit. *LREC 2020.*

Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT. *NAACL-HLT 2019.*

Kula, S., Choraś, M., & Kozik, R. (2021). BERT-based fake news detection. *Springer LNNS.*

Rashkin, H. et al. (2017). Truth of varying shades. *EMNLP 2017.*

Silla, C. N., & Freitas, A. A. (2011). A survey of hierarchical classification. *Data Mining and Knowledge Discovery, 22*(1), 31–72.