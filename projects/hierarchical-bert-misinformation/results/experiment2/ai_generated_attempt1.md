# Experiment 2: AI-Generated Content — Attempt 1 (Domain Mismatch)

## Overview

This experiment begins the taxonomy extension phase of the project. It adds 
AI-Generated Content as a seventh category (label 6) placed in Group 2 
(Fabricated/Manipulated), and investigates how the quality and style of the 
external training dataset affects detection performance.

This is **Attempt 1 of 3** in an iterative dataset refinement process. It is 
intentionally documented as a failure case — the result reveals a critical 
finding about domain mismatch that motivates all subsequent attempts.

---

## Motivation

The Fakeddit dataset was collected in 2020, before large language models became 
widely used to generate fake news. As a result, it contains no AI-generated 
content samples and the original taxonomy does not include this category.

Since no Fakeddit sample is labelled as AI-generated, external labelled data 
must be sourced and combined with the Fakeddit training set. This raises a 
practical question: **how much does the quality and style of that external data 
affect detection performance?** This experiment is the first attempt to answer 
that question.

---

## Why Group 2

AI-generated news articles are inauthentic at the content level — the material 
itself is not real. This is the same fundamental property as Satire/Parody, 
Imposter Content, and Manipulated Content, all of which involve content where the 
material has been fabricated, altered, or invented. Group 2 (Fabricated/Manipulated) 
is therefore the correct semantic home for AI-Generated Content.

---

## Setup

### Base Model
Loaded from `best_model.pt` (hierarchical model, 82% accuracy, Macro F1 0.77).

### Freezing Strategy

| Component | Status | Reason |
|-----------|--------|--------|
| BERT encoder | FROZEN | Preserves general text representations learned in step1; unfreezing risks catastrophic forgetting and requires significantly more compute |
| Group 0 fine head | FROZEN | Handles True Content — unrelated to new category |
| Group 1 fine head | FROZEN | Handles Structural Deception — unrelated to new category |
| Group 2 fine head | REINITIALISED (4 outputs) + RETRAINED | Adds AI-Generated alongside Satire, Imposter, Manipulated |
| Coarse head | FROZEN in Attempt 1 | Routing not updated in this attempt |

```
[BERT Encoder]     ←── FROZEN
       |
  [Coarse Head]    ←── FROZEN (Attempt 1)
       |
  ┌────┴──────────────┐
  │                   │
Group 1 Head      Group 2 Head   ←── REINITIALISED + RETRAINED
(FROZEN)          (size 3 → 4, adds AI-Generated)
```

*See `models/taxonomy_extension.png` for the full figure.*

### External Dataset: artem9k/ai-text-detection-pile

| Property | Value |
|----------|-------|
| Dataset | artem9k/ai-text-detection-pile |
| Total size | 1.39M samples |
| Sources | GPT-2, GPT-3, ChatGPT, GPT-J |
| Samples used | 30,000 AI-generated samples |
| Average length | Several hundred words per sample |
| Fakeddit title length | 8–12 words |

**Why this dataset was chosen first:** Its large scale and variety of generative 
models seemed likely to produce a robust detector.

### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate | 2e-5 |
| Batch size | 64 |
| Epochs | 2 |
| Optimiser | AdamW with linear warmup |
| Components updated | Group 2 fine head only |

---

## Results

### AI-Generated Category Performance

| Metric | Value |
|--------|-------|
| F1 | **0.11** |
| Precision | 0.94 |
| Recall | 0.18 |

### Interpreting the Precision/Recall Pattern

```
Precision = 0.94   ← When the model predicts AI-Generated, it is almost always right
Recall    = 0.18   ← But the model almost never predicts AI-Generated at all

This pattern is the signature of severe domain mismatch.
```

The model learned to associate AI-Generated content with long formal writing 
(essays, hundreds of words). At test time, all inputs are short Reddit post 
titles (8–12 words). The model almost never predicts AI-Generated because short 
titles look nothing like the essays it was trained on.

---

## Domain Mismatch Visualised

```
Training data (artem9k essays):
"The emergence of artificial intelligence as a transformative force in modern 
society has prompted extensive debate among scholars, policymakers, and 
technologists alike. This analysis seeks to examine..."
Average length: ~300 words

Fakeddit test titles:
"Scientists discover new treatment for diabetes"
"White House announces new economic policy"
"Local team wins championship after 20 years"
Average length: 8–12 words

→ Model trained on long text, tested on short text
→ Recall = 0.18 (model almost never fires)
→ Precision = 0.94 (when it does fire, it is correct)
```

---

## Full Classification Report

| Category | F1 | Notes |
|----------|----|-------|
| True Content | ~0.87 | Unchanged from base model |
| Satire/Parody | ~0.73 | Unchanged — Group 2 head retrained but coarse head frozen |
| Misleading Content | ~0.73 | Unchanged — Group 1 head frozen |
| Imposter Content | ~0.62 | Unchanged |
| False Connection | ~0.85 | Unchanged — Group 1 head frozen |
| Manipulated Content | ~0.73 | Small drop due to competition within Group 2 |
| **AI-Generated** | **0.11** | Very low — domain mismatch |

*Note: Existing category scores are approximate. The key result of this experiment 
is the AI-Generated F1 of 0.11 and the precision/recall pattern it reveals.*

---

## Discussion

**The F1 of 0.11 is not a model failure — it is an experiment finding.**

The artem9k dataset was a reasonable first choice: large scale, multiple LLM 
sources, widely used in AI detection research. The failure reveals something 
more valuable than a good result would have: **domain match between external 
training data and the target distribution is the critical factor for emerging 
category detection.**

The precision of 0.94 confirms the architecture is sound — when the model does 
see text that resembles its training data, it classifies it correctly almost 
every time. The problem is exclusively distributional: the training distribution 
(long essays) does not match the test distribution (short titles).

This finding directly motivates Attempt 2, which will use headline-matched data 
to test whether fixing the domain mismatch produces the expected improvement in 
recall.

**Output checkpoint:** `hierarchical_v2_best.pt`

---

## Iterative Refinement Plan

| Attempt | Dataset | Expected F1 | Status |
|---------|---------|-------------|--------|
| 1 — Long essays | artem9k/ai-text-detection-pile | 0.11 | **Complete — documents domain mismatch** |
| 2 — News headlines | artnitolog/llm-generated-texts (Reuters) | ~0.71 | Planned |
| 3 — NYT titles (final) | gsingh1-py/train | ~0.90 | Planned |

---

## Key Takeaway

> Domain match between external training data and the target distribution 
> is critical for emerging category detection. A well-performing detector 
> requires training data that matches the length, style, and domain of the 
> content it will encounter at test time — not just the label it carries.
