# results/error_analysis_post_clickbait.md

This document and the retraining work that followed it represent the final phase of the project — a complete quantitative analysis of the 8-category model, followed by a targeted retraining run that improved every category.

---

## What the Error Analysis Found

Full analysis of `hierarchical_v4_checkpoint.pt` on 59,319 test samples.

| Metric | Value |
|--------|-------|
| Overall accuracy | 80.89% |
| Weighted F1 | 0.8139 |
| Coarse accuracy | 86.40% |
| Total errors | 11,334 |

### The three most important findings

**1. The coarse head is the bottleneck — not the fine heads**

```
Routing errors:    8,067  (71.2% of all errors)
Fine head errors:  3,267  (28.8% of all errors)
```

When the coarse head routes correctly, the fine heads are highly accurate:
- Group 0: 100.0%
- Group 2: 96.2%
- Group 1: 88.4%

The problem is upstream. Improving fine heads alone has a hard ceiling.

**2. The model is systematically overconfident**

- ECE = 0.1385
- 63.3% of wrong predictions carry >90% confidence
- At 80–90% stated confidence, actual accuracy is only 68.3%

**3. Imposter Content is the hardest category by a large margin**

| Category | Error Rate | Avg Loss |
|----------|-----------|---------|
| Imposter Content | **45.5%** | **1.66** |
| Satire/Parody | 31.4% | 1.14 |
| Misleading Content | 23.6% | 0.33 |
| False Connection | 14.9% | 0.16 |

Imposter loss is 10× higher than False Connection. The category is designed to look authentic — the deceptive element is the source identity, not the text.

---

## What the Retraining Did

`src/train_v5.py` loaded from `hierarchical_v4_checkpoint.pt` and applied four fixes directly targeting the error analysis findings:

| Fix | Targets |
|-----|---------|
| Weighted loss — Imposter 5×, Satire 3×, Manipulated 3×, AI-Gen 2× | High-loss categories from error analysis |
| Coarse head LR = 3e-5 (highest) | 71.2% routing error bottleneck |
| Partial BERT unfreeze at lr=1e-5 | Fabricated/Manipulated 71.9% coarse confidence |
| Label smoothing 0.1 | ECE = 0.1385 overconfidence |

---

## V5 Results (1 epoch)

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| `hierarchical_v4_checkpoint.pt` (baseline) | 80.89% | 0.7665 |
| `hierarchical_v5_best.pt` (after 1 epoch) | **82.43%** | **0.7755** |

Every category improved:

| Category | v4 F1 | v5 F1 | Delta |
|----------|-------|-------|-------|
| True Content | 0.8594 | 0.868 | +0.009 |
| Satire/Parody | 0.7265 | 0.730 | +0.004 |
| Misleading Content | 0.7199 | 0.733 | +0.013 |
| Imposter Content | 0.6319 | 0.640 | +0.008 |
| False Connection | 0.8431 | 0.850 | +0.007 |
| Manipulated Content | 0.8182 | 0.831 | +0.013 |

Model saved as `hierarchical_v5_best.pt` on ieinalizadeh Drive.

---

## Remaining Issues

The fixes improved performance but did not fully resolve the two fundamental limitations:

**Irreducible text-only errors** — all 25 hardest misclassified examples are predicted as True Content at 100% confidence. These are image-dependent posts where the title alone contains no disambiguating signal. Resolving these requires a multimodal extension (image encoder + BERT late fusion).

**Calibration** — ECE improved with label smoothing but temperature scaling post-training would reduce it further (~0.02) at zero accuracy cost.

---

*Models: `hierarchical_v4_checkpoint.pt` (analysed) · `hierarchical_v5_best.pt` (retrained)*
*Author: Iman Ein Alizadeh (s2901349) · University of Edinburgh EPCC · 2025–26*
