# Experiment 3: Clickbait Detection Extension

## Overview

This experiment adds **Clickbait** as an eighth category (label 7) placed in Group 1 (Structural Deception), extending the taxonomy from 7 to 8 categories. Unlike the AI-Generated extension which required three iterative attempts to resolve domain mismatch, Clickbait achieves strong performance in a single run by applying all lessons learned from the AI-Generated experiments from the start.

---

## Motivation

Clickbait headlines are designed to maximise clicks through sensationalism, curiosity gaps, and emotional manipulation — often without being factually false. This makes them structurally distinct from other categories:

- They are not fabricated (unlike AI-Generated or Imposter Content)
- They are not factually wrong (unlike Misleading Content)
- The deception is in **form**, not content — the headline manipulates engagement, not belief

This places Clickbait naturally in **Group 1 — Structural Deception**, alongside Misleading Content and False Connection. All three categories deceive through structure and presentation rather than through fabrication.

---

## Why Group 1

| Category | Group | Deception mechanism |
|----------|-------|-------------------|
| Misleading Content | 1 — Structural Deception | Real facts, deceptive framing |
| False Connection | 1 — Structural Deception | Real media, mismatched headline |
| Clickbait | 1 — Structural Deception | Real or vague content, manipulative headline form |
| Satire/Parody | 2 — Fabricated/Manipulated | Invented content for humour |
| AI-Generated | 2 — Fabricated/Manipulated | Entirely fabricated by LLM |

Clickbait is structurally deceptive — not factually fabricated. Group 1 is the correct semantic home.

---

## Setup

### Base Model

Loaded from `hierarchical_v4_best.pt` (84% accuracy, Macro F1 0.81, AI-Generated F1 0.90).

### Freezing Strategy

| Component | Status | Reason |
|-----------|--------|--------|
| BERT encoder | Frozen | Preserves learned text representations |
| Group 0 fine head | Frozen | Handles True Content — unrelated |
| Group 2 fine head | Frozen | Handles Fabricated/Manipulated — unrelated |
| Coarse head | Frozen | Group 1 routing already correct |
| **Group 1 fine head** | **Reinitialised (3 outputs) + retrained** | Adds Clickbait alongside Misleading and False Connection |

```
[BERT Encoder]       <-- FROZEN
      |
 [Coarse Head]       <-- FROZEN
      |
 +----+---------------------+
 |                          |
Group 1 Head            Group 2 Head   <-- FROZEN
(size 2 -> 3)
REINITIALISED + RETRAINED
Adds Clickbait alongside
Misleading, False Connection
```

Group 1 fine label remapping:

| Category | Old fine label | New fine label |
|----------|---------------|---------------|
| Misleading Content | 0 | 0 |
| False Connection | 1 | 1 |
| Clickbait | -- | **2** (new) |

---

## Dataset

Clickbait samples were identified from existing Fakeddit Group 1 posts using `src/clickbait_dataset_builder.py`. Posts were scored using subreddit membership and 10 linguistic pattern regexes covering:

- Number-led headlines ("10 reasons why...")
- "You won't believe..." constructions
- Trailing ellipsis (...)
- ALL CAPS words (SHOCKING, BREAKING, VIRAL)
- Question mark endings
- Superlatives (best, worst, most, craziest)
- "Here's why / Find out / What happens when"

Posts scoring above 0.45 were relabelled as Clickbait (Category 7).

| Split | Clickbait samples | Misleading | False Connection | Total Group 1 |
|-------|------------------|-----------|-----------------|---------------|
| Train | ~4,500 | ~78,960 | ~168,072 | ~251,532 |
| Val | ~480 | ~8,340 | ~17,730 | ~26,550 |
| Test | ~475 | ~8,310 | ~17,640 | ~26,425 |

**Lessons from AI-Generated applied from Day 1:**
- Weighted cross-entropy loss: Clickbait 3x, Misleading 1x, False Connection 1x
- Oversampling of Clickbait to reach proportional representation
- No iterative attempts needed — domain is already correct (Fakeddit titles)

---

## Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Base model | `hierarchical_v4_best.pt` |
| Learning rate (BERT) | 1e-5 |
| Learning rate (heads) | 2e-5 |
| Batch size | 64 |
| Epochs | 2 |
| Optimiser | AdamW with linear warmup (10%) |
| Components updated | Group 1 fine head only |
| Class weights | Clickbait 3x, others 1x |

---

## Results

### Clickbait Category Performance

| Metric | Value |
|--------|-------|
| F1 | **0.83** |
| Precision | 0.85 |
| Recall | 0.81 |

### Full Classification Report

| Category | F1 | Change vs v4 |
|----------|----|-------------|
| True Content | 0.92 | Unchanged |
| Satire/Parody | 0.82 | Unchanged |
| Misleading Content | 0.77 | Unchanged |
| Imposter Content | 0.57 | Unchanged |
| False Connection | 0.80 | -0.01 (minor redistribution within Group 1) |
| Manipulated Content | 0.70 | Unchanged |
| AI-Generated Content | 0.90 | Unchanged |
| **Clickbait** | **0.83** | New category |
| **Macro F1** | **0.81** | Maintained |
| **Overall Accuracy** | **84%** | Maintained |

---

## Why Clickbait Worked First Time

Unlike AI-Generated Content — which required three attempts to resolve domain mismatch — Clickbait achieved F1 0.83 in a single run. Three reasons:

**1. No domain mismatch.** Clickbait samples come directly from Fakeddit subreddits. The training and test distributions are identical in length, style, and platform. There is no external dataset to align.

**2. Lessons applied from Day 1.** The AI-Generated experiments showed that class imbalance is fatal without correction. Weighted loss (3x) and oversampling were applied before training began — not discovered after a failure run.

**3. Linguistically distinct patterns.** Clickbait headlines have strong surface signals — number leads, superlatives, ALL CAPS, ellipsis, question marks — that BERT can learn reliably from headline-length text. The patterns are consistent and frequent enough that even a minority class is learnable.

---

## Precision/Recall Balance

```
Precision = 0.85   <- When the model predicts Clickbait, it is right 85% of the time
Recall    = 0.81   <- The model catches 81% of all actual clickbait headlines
```

This is a healthy, balanced result — no collapse in either direction. Compare to AI-Generated Attempt 1 (Precision 0.94, Recall 0.18) where the model almost never fired.

---

## Discussion

The contrast between the Clickbait and AI-Generated extensions illustrates the two failure modes for emerging category detection:

| Issue | AI-Generated Attempt 1 | Clickbait |
|-------|------------------------|-----------|
| Domain mismatch | Severe (essays vs titles) | None (Fakeddit to Fakeddit) |
| Class imbalance | Not addressed | Addressed from Day 1 |
| Result | F1 0.11 | F1 0.83 |

The remaining 17% recall gap (missed clickbait) is explained by two factors:

- **Overlap with Misleading Content** — some headlines are both misleading in framing and sensational in form; the model must choose one label
- **Subtle clickbait** — headlines that use soft engagement tactics (mild curiosity gaps, understated sensationalism) without triggering the strong surface patterns the model relies on

Both are expected limitations given the category definitions and are consistent with the broader finding that structural deception categories are harder to separate than fabrication categories.

---

## Output

- Checkpoint: `hierarchical_v4_clickbait.pt`
- Results saved to: `results/clickbait/`
- Dataset built by: `src/clickbait_dataset_builder.py`

---

## Complete Experiment Summary

| Experiment | Category added | F1 | Attempts needed |
|------------|---------------|-----|----------------|
| 1 | -- (hierarchical vs flat) | 0.77 vs 0.75 | 1 |
| 2a | AI-Generated | 0.11 | Attempt 1 of 3 |
| 2b | AI-Generated | 0.71 | Attempt 2 of 3 |
| 2c | AI-Generated | **0.90** | Attempt 3 of 3 |
| 3 | Clickbait | **0.83** | 1 |

---

## Key Takeaway

Clickbait detection succeeds in a single attempt because the training and test distributions are naturally aligned — no external data is needed. The key lesson from the AI-Generated experiments (fix domain and class imbalance before training, not after) was applied from the start. The result is a clean, balanced F1 of 0.83 with no precision/recall collapse.
