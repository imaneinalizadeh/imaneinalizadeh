# Experiment 1: Hierarchical vs Flat Baseline

## Overview

This experiment tests the core claim of the project: that a hierarchical two-stage BERT 
classifier outperforms a flat single-stage baseline on six-way fake news classification. 
Both models are trained under strictly identical conditions so that any difference in 
results can be attributed solely to the hierarchical architecture.

---

## Setup

| Parameter | Value |
|-----------|-------|
| Base model | bert-base-uncased |
| Training samples | 564,000 (full Fakeddit training set) |
| Test samples | 59,319 (standard Fakeddit public test split) |
| Learning rate | 2e-5 |
| Batch size | 64 |
| Max sequence length | 64 tokens |
| Optimiser | AdamW with linear warmup (10% steps) |
| Epochs | 2 |
| Input | clean_title column only |
| Random seed | 42 (also validated with seeds 123, 456) |

**Fair comparison controls:** Same backbone, same training data, same order, same seed, 
same evaluation set, same text input. Only the classification head differs.

---

## Results

### Per-Class F1 Scores

| Category | Group | Hierarchical F1 | Flat F1 | Difference |
|----------|-------|----------------|---------|------------|
| True Content | 0 — Authentic | 0.87 | 0.86 | **+0.01** |
| Satire/Parody | 2 — Fabricated | 0.73 | 0.69 | **+0.04** |
| Misleading Content | 1 — Structural | 0.73 | 0.72 | **+0.01** |
| Imposter Content | 2 — Fabricated | 0.62 | 0.58 | **+0.04** |
| False Connection | 1 — Structural | 0.85 | 0.85 | 0.00 |
| Manipulated Content | 2 — Fabricated | 0.83 | 0.80 | **+0.03** |

### Summary Metrics

| Metric | Hierarchical | Flat | Difference |
|--------|-------------|------|------------|
| Overall Accuracy | 82% | 81% | +1% |
| Macro F1 | 0.77 | 0.75 | **+0.02** |
| Weighted F1 | 0.82 | 0.81 | +0.01 |

---

## Statistical Reliability

Both models were trained three times with different random seeds (42, 123, 456) to 
confirm results are stable and not an artefact of a single initialisation.

| Model | Macro F1 Mean | Std Dev |
|-------|--------------|---------|
| Flat Baseline | 0.75 | ±0.006 |
| Hierarchical | 0.77 | ±0.005 |

Low standard deviations confirm both models are stable across seeds. The hierarchical 
model outperformed the flat baseline in all three runs. A paired t-test confirms the 
macro F1 difference is statistically significant (p < 0.05).

---

## Architecture Diagram

```
Input Text (post title)
        |
   [BERT Encoder]          ← shared weights, bert-base-uncased
        |
   [CLS] token (768d)
        |
  ┌─────┴─────┐
  │  Coarse   │  ──→  Group 0 / Group 1 / Group 2
  │   Head    │
  └───────────┘
        |
  ┌─────┴──────────────────────────┐
  │             │                  │
Group 0 Head  Group 1 Head    Group 2 Head
(1 output)    (2 outputs)     (3 outputs)
True Content  Misleading      Satire
              False Conn.     Imposter
                              Manipulated
```

*See `models/hierarchical_bert_architecture.png` for the full figure.*

---

## Discussion

The hierarchical model outperforms the flat baseline on five of six categories. The 
largest gains are on **Satire/Parody (+0.04)** and **Imposter Content (+0.04)** — both 
minority classes in Group 2. This is exactly what the hierarchical design predicts.

In the flat model, these small classes must compete against all six categories at once, 
including dominant classes like True Content (39.4% of training data) and False 
Connection (29.8%). In the hierarchical model, the coarse head first routes samples into 
Group 2, and only then does the fine head distinguish between Satire, Imposter, and 
Manipulated. The decision space is smaller and more semantically coherent at each stage, 
which directly benefits underrepresented classes.

**False Connection shows no improvement (0.85 both models).** This is expected — False 
Connection is the dominant class within Group 1. The flat model already has sufficient 
signal to classify it well without hierarchical assistance.

**The overall accuracy gain of 1% is modest and expected.** Both models handle the 
dominant classes well, so the improvement is concentrated in the minority classes. This 
is precisely why macro F1 is the more informative metric for imbalanced datasets — it 
weights all classes equally and shows a consistent +0.02 improvement that accuracy alone 
would understate.

**These results confirm the primary hypothesis:** a hierarchical architecture achieves 
higher classification performance than a flat baseline, with the greatest gains on 
minority and semantically related categories.

---

## Per-Class Breakdown (Hierarchical Model)

| Category | Precision | Recall | F1 | Support |
|----------|-----------|--------|-----|---------|
| True Content | 0.87 | 0.87 | 0.87 | 23,507 |
| Satire/Parody | 0.78 | 0.70 | 0.73 | 3,514 |
| Misleading Content | 0.73 | 0.72 | 0.73 | 11,297 |
| Imposter Content | 0.83 | 0.49 | 0.62 | 1,224 |
| False Connection | 0.82 | 0.88 | 0.85 | 17,472 |
| Manipulated Content | 0.83 | 0.82 | 0.83 | 2,305 |
| **Macro Average** | **0.81** | **0.75** | **0.77** | 59,319 |
| **Weighted Average** | **0.82** | **0.82** | **0.82** | 59,319 |

---

## Key Takeaways

- Hierarchical model outperforms flat baseline on 5 of 6 categories
- Largest gains on minority Group 2 classes (Satire +0.04, Imposter +0.04)
- Macro F1 improvement of 0.02 is statistically significant (p < 0.05) across 3 seeds
- Result confirms the value of modelling semantic group structure explicitly
- Imposter Content remains the weakest class (F1 0.62) due to severe class imbalance (1,224 test samples vs 23,507 for True Content) — targeted oversampling or weighted loss is identified as future work
