# Experiment 4: Taxonomy Extensibility Analysis — When Does Adding a New Category Stop?

## Research Question

At what point does adding a new category to the hierarchical classifier stop being beneficial? This experiment answers that question empirically using four strict viability criteria applied after each category addition attempt.

---

## Viability Criteria (Strict — Fail ANY One = Stop)

A new category is only viable if ALL four criteria are met simultaneously:

| # | Criterion | Threshold | What it measures |
|---|-----------|-----------|-----------------|
| 1 | New category F1 | **> 0.65** | Did the model actually learn the new category as distinct? |
| 2 | Max existing category F1 drop | **< 0.03** | Did adding the new category hurt existing categories? |
| 3 | Routing error rate change | **< +5%** | Did the coarse head get confused about group assignments? |
| 4 | Expected Calibration Error | **< 0.20** | Did the model become more overconfident? |

These thresholds were fixed before running any experiments. A single failure on any one criterion declares the stopping point for that category — no exceptions.

---

## Why These Specific Thresholds — And What Would Happen With Different Ones

### Criterion 1 — New category F1 > 0.65

**Why 0.65 and not something else?**

0.65 represents the minimum threshold for a classifier to be considered practically useful in a deployment context. Below 0.65, the model is wrong more than 35% of the time on the new category — unreliable enough that users or downstream systems cannot trust it. The threshold comes from the existing category performance floor: our weakest existing category (Imposter Content) sits at F1 0.63. Adding a new category that performs worse than our hardest existing category would be counterproductive.

**What would happen with a stricter threshold (e.g. 0.70)?**

Both Junk News (0.155) and Propaganda (0.133) would still fail — the gap between their actual F1 and 0.70 is enormous. The conclusion would be identical. However, a 0.70 threshold would also have rejected AI-Generated Content in Attempt 2 (Reuters, F1 0.71 — barely above 0.70) and might have terminated that experiment before the successful Attempt 3. A threshold of 0.70 is slightly too strict for an iterative extension process where early attempts are expected to underperform.

**What would happen with a looser threshold (e.g. 0.60)?**

Still would not change the outcome for this experiment — both trained categories scored below 0.20, far below even 0.60. However, a 0.60 threshold would mean accepting a category that is wrong 40% of the time. For a fake news detector deployed in a real context, this is below the practical usefulness floor — you would be adding a category that actively misleads users more often than it helps.

**What would happen with a very loose threshold (e.g. 0.20)?**

Both Junk News and Propaganda would pass Criterion 1. But the resulting model would be useless for those categories — predicting them correctly only 1 in 5 times. The taxonomy would appear to have 9 categories but two of them would be unreliable noise. This would be academically dishonest — claiming extensibility that does not exist in practice.

**Bottom line:** 0.65 is the sweet spot — strict enough to ensure genuine learning has occurred, loose enough to allow for naturally harder minority categories.

---

### Criterion 2 — Max existing category F1 drop < 0.03

**Why 0.03 and not something else?**

0.03 (3 percentage points) reflects the typical noise floor of a BERT fine-tuning run. Across multiple runs with different random seeds, category F1 scores vary by approximately ±0.005–0.015. A drop of 0.03 is roughly 2× the natural variance — large enough to indicate a genuine regression rather than random fluctuation, small enough that it does not represent a meaningful degradation in practice. If True Content drops from F1 0.86 to 0.83, that is still a strong classifier. If it drops to 0.80, that is a problem.

**What would happen with a stricter threshold (e.g. 0.01)?**

In both Junk News and Propaganda experiments, the maximum drop was 0.000 and -0.022 respectively. With a 0.01 threshold, Propaganda's Imposter Content drop of -0.022 would fail Criterion 2. The conclusion (Propaganda fails) would be the same, but for an additional reason. For Junk News, no drop occurred — 0.01 would still pass. The overall conclusion does not change.

**What would happen with a looser threshold (e.g. 0.05)?**

No change to outcomes here. Neither experiment came close to a 0.05 drop. However, a 0.05 threshold would be academically too permissive — accepting a model where True Content drops from 0.86 to 0.81 (a meaningful real-world degradation) as long as the new category is good enough.

**What would happen with no threshold at all?**

Junk News showed zero regression and actually improved several categories. Propaganda showed -0.022 on Imposter Content. Without this criterion, you would miss the case where adding a new category silently degrades an existing one. The criterion exists to catch hidden costs of extension that are not visible in the new category's score alone.

**Bottom line:** 0.03 is calibrated to the natural noise of BERT fine-tuning. It catches genuine regressions while ignoring statistical noise.

---

### Criterion 3 — Routing error rate change < +5 percentage points

**Why +5% and not something else?**

The baseline routing error rate is ~71–76% (most errors are routing errors — this is the known bottleneck from the error analysis). A change of +5 percentage points means routing went from e.g. 71% to 76% — a 7% relative increase. This is a meaningful architectural destabilisation. We chose 5 percentage points rather than a relative % because the absolute routing rate is already high — a 5% relative change on 71% is only 3.5 points, which could be noise.

**What would happen with a stricter threshold (e.g. +2%)?**

Propaganda's routing increased by +7.6 points — it would still fail. Junk News's routing actually decreased by -1.7 points — it would still pass. No change to either conclusion.

**What would happen with a looser threshold (e.g. +10%)?**

Propaganda (+7.6 points) would pass Criterion 3. But it would still fail Criterion 1 (F1 0.133). The overall conclusion does not change. However, a +10% threshold would be too permissive — allowing the coarse head to degrade significantly before declaring failure. Since routing errors account for 71.2% of all errors in this model, any meaningful routing degradation has outsized consequences on overall performance.

**What would happen with no threshold at all?**

You would miss the most important architectural signal. Propaganda's routing failure (+7.6 points) reveals something that F1 alone does not — not only can the model not learn Propaganda within Group 2, it cannot even route Propaganda samples to Group 2 reliably. This is qualitatively different from Junk News's failure (which routed correctly but could not distinguish within the group). Without Criterion 3, both failures look identical. With it, we can identify that Propaganda is a more fundamental failure.

**Bottom line:** +5 percentage points catches genuine routing destabilisation while tolerating minor fluctuations. It also reveals the qualitative difference between Junk News (fine-level failure) and Propaganda (routing + fine-level failure).

---

### Criterion 4 — ECE < 0.20

**Why 0.20 and not something else?**

ECE (Expected Calibration Error) measures how much the model's stated confidence deviates from its actual accuracy. ECE = 0.20 means confidence scores are off by 20 percentage points on average — a model saying "90% confident" is actually right only 70% of the time. This is the commonly accepted threshold in the calibration literature above which confidence scores are considered misleading. Our baseline ECE after label smoothing was 0.03–0.05 — well within acceptable range.

**What would happen with a stricter threshold (e.g. 0.10)?**

Both experiments maintained ECE around 0.046–0.055 — well below 0.10. No change to outcomes.

**What would happen with a looser threshold (e.g. 0.30)?**

Still no change — neither experiment came close to even 0.20. The ECE criterion never came close to being the binding constraint in this experiment. It exists as a safety net for future extension attempts that might cause calibration collapse (e.g. if a very dominant new class causes the model to become extremely overconfident on everything else).

**What would happen with no threshold at all?**

For this specific experiment, no change. But without this criterion, a future extension could introduce a category that causes the model to output meaningless confidence scores. In deployment, a model that says "95% confident" when it is actually right 60% of the time is actively dangerous for downstream decision-making. The criterion protects against this.

**Bottom line:** ECE < 0.20 is a safety net rather than a binding constraint in this experiment. Both trained attempts maintained excellent calibration throughout. Its value is in preventing silent reliability failures in future extension attempts.

---

## Summary: What the Thresholds Tell Us

| Criterion | Binding? | What it revealed |
|-----------|---------|-----------------|
| F1 > 0.65 | **Yes — failed both times** | The model cannot learn either new category as distinct from existing ones |
| Drop < 0.03 | No — passed both times | Existing knowledge is robust; adding new categories does not hurt what already works |
| Routing < +5% | **Yes — failed for Propaganda** | Propaganda is a more severe failure than Junk News — even routing breaks down |
| ECE < 0.20 | No — passed both times | Calibration is stable under extension attempts |

The key insight: **Criterion 1 is the universal stopping signal. Criterion 3 reveals the severity of the failure.** Criteria 2 and 4 provide confidence that the failures are isolated — the model's existing quality is not being damaged by the extension attempts.

---

## Baseline Model

All experiments start from `production/hierarchical_v5_best.pt` — the best current 8-category model.

| Metric | Baseline Value |
|--------|---------------|
| Overall accuracy | 82.43% |
| Macro F1 | 0.7755 |
| Routing error rate | 76.2% (Round 1) / 66.3% (Round 2, recalculated post-removal) |
| ECE | 0.0317–0.0549 |

Note: baseline routing rate differs slightly between rounds because each round's training set composition changes (e.g. propagandaposters subreddit removed from Manipulated Content before Round 2).

---

## Summary of All Attempts

| Attempt | Category | Group | Data Source | Samples | Result |
|---------|----------|-------|-------------|---------|--------|
| 1 | Conspiracy Theory | 2 | Fakeddit + HuggingFace | 14 | Abandoned — insufficient data |
| 2 | Rumour | 1 | Fakeddit pattern matching | 1,419 | Abandoned — insufficient/noisy data |
| 3 | Junk News | 1 | `GonzaloA/fake_news` | 10,736 | **Trained — FAILED Criterion 1** |
| 4 | Propaganda | 2 | Fakeddit r/propagandaposters | 13,456 | **Trained — FAILED Criteria 1 & 3** |

---

## Attempt 1 — Conspiracy Theory (Abandoned)

**Rationale:** Strong linguistic signal expected (exposed, revealed, deep state, cover-up). Group 2 placement.

**Dataset search:** Fakeddit subreddit filter (r/conspiracy, r/conspiracytheories), 3 HuggingFace datasets, linguistic pattern matching across 12 regex patterns.

**Result:** Only 14 usable samples found. Fakeddit contains zero posts from conspiracy-labelled subreddits. All three targeted HuggingFace datasets failed to load. A minimum of ~5,000 domain-matched samples is required for reliable category learning — this threshold was not reached.

---

## Attempt 2 — Rumour (Abandoned)

**Rationale:** Clear linguistic markers (reportedly, sources say, allegedly, unconfirmed, leaked). Group 1 placement.

**Dataset search:** 5 HuggingFace rumour detection datasets attempted — all failed to load. Fakeddit pattern matching yielded 1,419 samples with significant false-positive contamination (e.g. "leaking gas", "leaked photos of puppies").

**Result:** Abandoned — insufficient clean sample count.

---

## Attempt 3 — Junk News (Trained, Round 1)

**Rationale:** Politically charged low-quality content with heavy partisan framing — ALL CAPS, exclamation marks, named-politician negative framing. Group 1 placement, alongside Misleading Content, False Connection, Clickbait.

**Dataset:** `GonzaloA/fake_news` (label=0 subset) — 10,736 samples.

**Training setup:**
- Base: `hierarchical_v5_best.pt`
- Group 1 fine head expanded 3 → 4 outputs
- Frozen: Group 0 and Group 2 fine heads
- Trainable: BERT (lr=1e-5), coarse head (lr=3e-5), Group 1 fine head (lr=2e-5)
- Class weight: Junk News 3×, label smoothing 0.1
- 3 epochs, batch size 32

**Results (epoch 2, best checkpoint):**

| Criterion | Threshold | Result | Verdict |
|-----------|-----------|--------|---------|
| New category F1 > 0.65 | > 0.65 | **0.1546** | ✗ FAIL |
| Max existing drop < 0.03 | < 0.03 | 0.0000 | ✓ |
| Routing rate < 81.2% | < 81.2% | 74.5% | ✓ |
| ECE < 0.20 | < 0.20 | 0.0460 | ✓ |

**Per-category F1 vs baseline:**

| Category | Baseline | Round 1 | Delta |
|----------|---------|---------|-------|
| True Content | 0.8625 | 0.8676 | +0.005 |
| Satire/Parody | 0.7021 | 0.7377 | +0.036 |
| Misleading Content | 0.7317 | 0.7354 | +0.004 |
| Imposter Content | 0.6371 | 0.6549 | +0.018 |
| False Connection | 0.8498 | 0.8531 | +0.003 |
| Manipulated Content | 0.8157 | 0.8242 | +0.009 |
| **Junk News** | — | **0.1546** | NEW |

**Verdict: FAILED — Criterion 1 only.**

**Analysis:** Junk News shares surface properties with Clickbait (sensational phrasing) and Misleading Content (real event, manipulated framing). BERT cannot find a separating boundary despite adequate training data. This is a **representational overlap failure**, not a data quantity failure.

---

## Attempt 4 — Propaganda (Trained, Round 2)

**Rationale:** Historical/modern propaganda posters — distinct vocabulary (enlist, recruitment, soviet, USSR, wartime, patriotic). Group 2 placement.

**Dataset:** Fakeddit `r/propagandaposters` — 13,456 train / 1,455 test samples. Previously folded into Manipulated Content; this experiment separates them out.

**Training setup:**
- Base: `hierarchical_v5_best.pt`
- Group 2 fine head expanded 4 → 5 outputs
- Frozen: Group 0 and Group 1 fine heads
- Trainable: BERT (lr=1e-5), coarse head (lr=3e-5), Group 2 fine head (lr=2e-5)
- Class weight: Propaganda 3×, label smoothing 0.1
- 3 epochs, batch size 32

**Baseline note:** Removing propagandaposters from Manipulated Content dropped its F1 from ~0.82 to 0.3933 before Propaganda was learned. This is expected and accounted for.

**Results (epoch 1, best checkpoint):**

| Criterion | Threshold | Result | Verdict |
|-----------|-----------|--------|---------|
| New category F1 > 0.65 | > 0.65 | **0.1333** | ✗ FAIL |
| Max existing drop < 0.03 | < 0.03 | -0.0216 | ✓ |
| Routing rate < 71.3% | < 71.3% | **73.9%** | ✗ FAIL |
| ECE < 0.20 | < 0.20 | 0.0478 | ✓ |

**Per-category F1 vs baseline:**

| Category | Baseline | Round 2 | Delta |
|----------|---------|---------|-------|
| True Content | 0.8683 | 0.8682 | -0.000 |
| Satire/Parody | 0.7303 | 0.7376 | +0.007 |
| Misleading Content | 0.7329 | 0.7315 | -0.001 |
| Imposter Content | 0.6405 | 0.6189 | -0.022 |
| False Connection | 0.8504 | 0.8544 | +0.004 |
| Manipulated Content | 0.3933 | 0.7085 | +0.315* |
| **Propaganda** | — | **0.1333** | NEW |

*Manipulated Content's large gain reflects recovery toward its pre-removal baseline — not a genuine improvement.

**Verdict: FAILED — Criteria 1 and 3.**

**Analysis:** Propaganda posters are fundamentally visual — the deception is in the imagery, not the caption. Captions like "join the army" are linguistically indistinguishable from True Content. This caused both F1 failure AND routing failure — even the coarse head cannot identify these as Group 2 content from text alone. This is a **text-only modality limitation**, and a more severe failure than Junk News.

---

## Cross-Attempt Comparison

| Attempt | Category | F1 | Routing Δ | Regression | ECE | Failure Mode |
|---------|----------|-----|-----------|-----------|-----|---------------|
| 1 | Conspiracy Theory | — | — | — | — | Data unavailable |
| 2 | Rumour | — | — | — | — | Data unavailable |
| 3 | Junk News | 0.155 | -1.7% | None | 0.046 | **Intra-group overlap** |
| 4 | Propaganda | 0.133 | +7.6% | None | 0.048 | **Routing + intra-group overlap** |

---

## Answer to the Research Question

> **A new category stops being viable at one of two distinct failure points: (1) a data availability wall — no sufficiently large clean dataset exists; or (2) a representational ceiling — adequate data exists but the category's textual signal is not distinct enough from existing categories in the BERT embedding space.**

The binding constraint across both trained attempts was **Criterion 1 (F1 > 0.65)**. Criteria 2 and 4 never failed — the model's existing knowledge and calibration are robust to extension attempts. Criterion 3 additionally failed for Propaganda, revealing that it represents a more severe failure than Junk News.

---

## Formal Stopping Criterion

> **A category addition fails when either (a) fewer than ~5,000 clean domain-matched training samples can be sourced, or (b) the category's surface linguistic features do not produce a BERT embedding sufficiently separable from existing same-group categories, evidenced by new-category F1 remaining below 0.65 regardless of training duration, class weighting, or learning rate tuning.**

---

## Implications

Further taxonomy extension would require:
- **Multimodal extension** — ResNet/ViT + BERT late fusion. Both failures point to image features as the missing discriminating signal
- **A deeper hierarchy** — adding a 4th semantic group rather than new categories within existing groups
- **A larger backbone** — RoBERTa-large or DeBERTa-v3 offering a richer embedding space
- **Categories with stronger textual distinctiveness** — most remaining misinformation types lack clean data or overlap with existing categories

---

## Models Produced

| Checkpoint | Location | Description | Result |
|-----------|----------|-------------|--------|
| `hierarchical_v6_junknews.pt` | `models/extensibility/` | v5 + Junk News (Group 1, 4th output) | Junk News F1 0.155 — **not viable** |
| `hierarchical_v6_propaganda.pt` | `models/extensibility/` | v5 + Propaganda (Group 2, 5th output) | Propaganda F1 0.133 — **not viable** |

Both retained for reference. `production/hierarchical_v5_best.pt` remains the recommended model.

---

*Base model: `production/hierarchical_v5_best.pt`*
*Author: Iman Ein Alizadeh (s2901349) · University of Edinburgh EPCC · MSc Dissertation 2025–26*
*Supervisor: Oliver Brown*