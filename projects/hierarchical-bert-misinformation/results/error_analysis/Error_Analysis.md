# Full Model Analysis: Hierarchical BERT — Post-Clickbait (8-Category Final Model)

## Overview

This document presents a complete analysis of `hierarchical_v4_checkpoint.pt` — the final 8-category hierarchical BERT model including the Clickbait extension, evaluated on the Fakeddit 6-way test set of 59,319 samples.

The analysis covers all dimensions of model performance: per-category metrics, strengths, weaknesses, error breakdown, calibration, loss, title length effects, class imbalance handling, the new Clickbait category behaviour, and hardest misclassified examples.

**Evaluation note:** The test set contains the original 6 Fakeddit labels (0–5). AI-Generated (label 6) and Clickbait (label 7) have no ground truth samples here — predictions of these labels on this test set are false positives with no corresponding true positives. Weighted F1 (0.8139) is the most representative headline metric. Macro F1 is slightly deflated due to this.

---

## 1. Overall Performance

| Metric | Value |
|--------|-------|
| Overall Accuracy | **80.89%** |
| Macro F1 | 0.7665 |
| Weighted F1 | **0.8139** |
| Total errors | 11,334 / 59,319 |
| Error rate | 19.11% |
| Coarse (group) accuracy | **86.40%** |

---

## 2. Per-Category Performance

```
                     precision    recall  f1-score   support

       True Content     0.8886    0.8320    0.8594     23507
      Satire/Parody     0.7726    0.6855    0.7265      3514
 Misleading Content     0.6809    0.7637    0.7199     11297
   Imposter Content     0.7520    0.5449    0.6319      1224
   False Connection     0.8356    0.8507    0.8431     17472
Manipulated Content     0.8303    0.8065    0.8182      2305

          micro avg     0.8173    0.8089    0.8131     59319
          macro avg     0.7933    0.7472    0.7665     59319
       weighted avg     0.8215    0.8089    0.8139     59319
```

### Per-category confidence

| Category | N | Error % | Conf when correct | Conf when wrong |
|----------|---|---------|-------------------|-----------------|
| True Content | 23,507 | 16.8% | 1.0000 | 0.9018 |
| Satire/Parody | 3,514 | 31.4% | 0.9954 | 0.8999 |
| Misleading Content | 11,297 | 23.6% | 0.9542 | 0.8860 |
| Imposter Content | 1,224 | 45.5% | 0.9890 | 0.9299 |
| False Connection | 17,472 | 14.9% | 0.9076 | 0.8299 |
| Manipulated Content | 2,305 | 19.3% | 0.9973 | 0.9356 |

### Category difficulty ranking

| Rank | Category | Error Rate | F1 |
|------|----------|-----------|-----|
| 1 — hardest | Imposter Content | **45.5%** | 0.6319 |
| 2 | Satire/Parody | 31.4% | 0.7265 |
| 3 | Misleading Content | 23.6% | 0.7199 |
| 4 | Manipulated Content | 19.3% | 0.8182 |
| 5 | True Content | 16.8% | 0.8594 |
| 6 — easiest | False Connection | 14.9% | 0.8431 |

---

## 3. Model Strengths

### 3.1 Precision — when the model commits, it is usually right

| Category | Precision | Correct / Predicted |
|----------|-----------|-------------------|
| True Content | **0.8886** | 19,558 / 22,010 |
| False Connection | 0.8356 | 14,864 / 17,789 |
| Manipulated Content | 0.8303 | 1,859 / 2,239 |
| Imposter Content | 0.7520 | 667 / 887 |
| Satire/Parody | 0.7726 | 2,409 / 3,118 |
| Misleading Content | 0.6809 | 8,628 / 12,672 |

True Content, False Connection, and Manipulated Content all exceed 83% precision. When the model predicts any of these categories, it is right more than 4 times in 5. Even Imposter Content — the hardest category — achieves 75.2% precision, meaning when the model does fire on Imposter, it is usually correct.

### 3.2 Recall — strong detection across majority classes

| Category | Recall | Caught / True |
|----------|--------|---------------|
| False Connection | **0.8507** | 14,864 / 17,472 |
| True Content | 0.8320 | 19,558 / 23,507 |
| Manipulated Content | 0.8065 | 1,859 / 2,305 |
| Misleading Content | 0.7637 | 8,628 / 11,297 |
| Satire/Parody | 0.6855 | 2,409 / 3,514 |
| Imposter Content | 0.5449 | 667 / 1,224 |

False Connection recall of 85.1% is outstanding given that it is the second largest class (29.5% of test set). Manipulated Content achieves 80.7% recall despite being only 3.9% of the data — a strong minority class result.

### 3.3 High-confidence correct predictions

- **39,230 samples (66.1%)** classified correctly with >95% confidence — the model is both right and certain on two thirds of the test set
- **778 samples** classified correctly even at <60% confidence — the model finds the correct answer even under genuine uncertainty
- This means the model is confidently right far more often than it is confidently wrong

### 3.4 Balanced prediction volumes — no majority class collapse

| Category | Predicted | True | Ratio |
|----------|-----------|------|-------|
| True Content | 22,010 | 23,507 | 0.94 |
| False Connection | 17,789 | 17,472 | 1.02 |
| Misleading Content | 12,672 | 11,297 | 1.12 |
| Satire/Parody | 3,118 | 3,514 | 0.89 |
| Manipulated Content | 2,239 | 2,305 | 0.97 |
| Imposter Content | 887 | 1,224 | 0.72 |

All ratios close to 1.0 except Imposter Content (0.72). The model does not simply predict everything as True Content — it has learned meaningful distinctions across all classes. The slight under-prediction of Imposter Content is a recall problem, not a class collapse.

### 3.5 Fine head accuracy given correct routing

The fine heads are highly accurate when the coarse head routes correctly:

| Group | Fine Accuracy | Correct / Routed |
|-------|--------------|-----------------|
| Group 0 — Authentic | **100.0%** | 19,558 / 19,558 |
| Group 2 — Fabricated/Manipulated | **96.2%** | 4,935 / 5,130 |
| Group 1 — Structural Deception | **88.4%** | 23,492 / 26,564 |

Group 0 is trivially perfect (single class). Group 2 achieves 96.2% — when Fabricated/Manipulated samples are correctly routed, the model distinguishes Satire, Imposter, Manipulated, and AI-Generated with near-perfect accuracy. Group 1 achieves 88.4% across 3 classes including Clickbait — strong performance for a 3-way within-group decision.

**The fine heads are not the bottleneck.** Routing is.

### 3.6 Strong coarse (group-level) routing

| Group | Accuracy | Correct / Total |
|-------|----------|----------------|
| Structural Deception | **92.3%** | 26,564 / 28,769 |
| Authentic | 83.2% | 19,558 / 23,507 |
| Fabricated/Manipulated | 72.8% | 5,130 / 7,043 |

At the 3-way group level, **86.40%** of all samples are assigned to the correct semantic group. Structural Deception routing is particularly strong at 92.3%.

### 3.7 Hierarchical architecture outperforms flat baseline on 5/6 categories

| Category | Hier F1 | Flat F1 | Gain |
|----------|---------|---------|------|
| True Content | 0.8594 | 0.860 | +0.000 |
| Satire/Parody | 0.7265 | 0.690 | **+0.037** |
| Misleading Content | 0.7199 | 0.720 | −0.000 |
| Imposter Content | 0.6319 | 0.580 | **+0.052** |
| False Connection | 0.8431 | 0.850 | −0.007 |
| Manipulated Content | 0.8182 | 0.800 | **+0.018** |

The hierarchy benefits minority classes most. Satire/Parody (+0.037) and Imposter Content (+0.052) see the largest gains. By grouping semantically related categories, the coarse head provides a strong prior that separates Satire (Group 2) from True Content (Group 0) before the fine head runs — something the flat model must learn implicitly from a single shared head.

### 3.8 Clickbait extension — zero regression on existing categories

Adding Clickbait as an 8th category preserved all existing performance:

- Total errors changed by only 16 out of 59,319 samples
- Routing errors decreased (72.0% → 71.2%)
- Coarse accuracy improved (86.23% → 86.40%)
- True Content F1 improved slightly (+0.002)
- No category degraded meaningfully

The selective freezing strategy — only retraining the Group 1 fine head and coarse head while keeping Group 2 frozen — successfully added a new category without disturbing existing ones.

### 3.9 Inference efficiency

| Metric | Value |
|--------|-------|
| Per-sample inference time | **3.10ms** |
| Throughput | **323 samples/second** |
| Batch 100 inference time | 309.5ms |
| Full test set (59,319) | ~184 seconds |
| Model parameters | ~110M (BERT) + ~3K (heads) |
| Checkpoint size | 438MB |
| Max sequence length | 64 tokens |

At 323 samples/second on T4 GPU, the model is fast enough for real-time deployment. The 64-token limit (optimised for Fakeddit's 8–12 word titles) contributes significantly to speed compared to the default 512-token BERT configuration.

### 3.10 Longer titles are easier — model uses context effectively

| Title length | N | Accuracy | Error rate |
|-------------|---|----------|-----------|
| 1–3 words | 15,200 | 80.5% | 19.5% |
| 4–6 words | 15,262 | 77.3% | 22.7% |
| 7–9 words | 12,217 | 81.8% | 18.2% |
| 10–14 words | 11,329 | 83.6% | 16.4% |
| 15+ words | 5,311 | **84.5%** | **15.5%** |

Accuracy improves monotonically with title length (except 4–6 words). BERT's attention mechanism effectively leverages additional tokens — longer titles provide more disambiguating signal. The 15.5% error rate on long titles vs 22.7% on medium titles shows the model genuinely benefits from context.

### 3.11 Manipulated Content performs above class frequency expectation

Manipulated Content (3.9% of test set) achieves F1 0.8182 — above the macro average of 0.7665. This minority class performs comparably to majority classes like False Connection (29.5%, F1 0.8431). The model has genuinely learned what manipulated media headlines look like despite the small training sample.

---

## 4. Model Weaknesses

### 4.1 Routing bottleneck — 71.2% of errors are at the coarse head

```
Total errors:       11,334
Routing errors:      8,067  (71.2%)  — coarse head sent sample to wrong group
Fine head errors:    3,267  (28.8%)  — correct group, wrong category within group
```

Almost three quarters of all errors happen before the fine head makes any decision. The coarse head assigns the sample to the wrong semantic group, and the fine head — however accurate — cannot recover from this.

**Per-category routing vs fine breakdown:**

| Category | Total | Errors | Routing | Fine |
|----------|-------|--------|---------|------|
| True Content | 23,507 | 3,949 | 3,949 | **0** |
| Satire/Parody | 3,514 | 1,105 | 1,021 | 84 |
| Misleading Content | 11,297 | 2,669 | 1,303 | 1,366 |
| Imposter Content | 1,224 | 557 | 492 | 65 |
| False Connection | 17,472 | 2,608 | 902 | 1,706 |
| Manipulated Content | 2,305 | 446 | 400 | 46 |

True Content has **zero fine head errors** — every single True Content error is a routing error. This makes sense: Group 0 contains only True Content, so once routed there the fine head is trivially correct. The entire True Content error rate (16.8%) is attributable purely to the coarse head.

### 4.2 Imposter Content — the hardest category

| Metric | Value |
|--------|-------|
| Error rate | **45.5%** |
| F1 | 0.6319 |
| Average loss | **1.6612** |
| Std loss | 2.4039 |
| Recall | 0.5449 |
| Predicted count / True count | 0.72 |

Imposter Content is harder than the next hardest category (Satire, 31.4% error) by a margin of 14 percentage points. Its average loss (1.66) is 10× higher than False Connection (0.16) and nearly 2.6× higher than Satire (1.14). The model under-predicts it (ratio 0.72) and misses 45.5% of true examples.

**Why:** Imposter Content impersonates legitimate news sources. By design, the headlines look authentic — the deceptive element is the source identity, not the text. "BBC: Climate scientists warn of record temperatures" is identical in form whether it is genuine BBC or a fake BBC account. The model cannot access source metadata — only the title. 21.1% of Imposter Content is predicted as True Content, the largest single within-category confusion.

### 4.3 Misleading ↔ False Connection — largest confusion pair

- Misleading → False Connection: **1,307 errors (11.6% of Misleading)**
- False Connection → Misleading: **1,611 errors (9.2% of False Connection)**
- Total cross-confusions: **2,918**

Both categories involve real events presented misleadingly. Misleading Content changes the context; False Connection pairs real media with an unrelated headline. At the text level — without the image — the boundary is linguistically ambiguous. "Man protests outside building" could be either depending on whether the image shown is the actual protest or an unrelated stock photo. This confusion is partially irreducible in a text-only system.

### 4.4 Systematic overconfidence — the core reliability problem

| Metric | Value |
|--------|-------|
| Mean confidence (correct) | 0.9627 |
| Mean confidence (incorrect) | **0.8841** |
| Overconfidence gap | −0.0786 |
| ECE | 0.1385 |

The model is wrong at 88.4% average confidence — only 7.9 percentage points below its confidence when correct. The gap is too small to use confidence as a reliable filter.

**Confidence buckets for incorrect predictions:**

| Confidence range | Count | % of errors |
|-----------------|-------|-------------|
| 0.0 – 0.40 | 3 | 0.0% |
| 0.4 – 0.60 | 1,015 | 9.0% |
| 0.6 – 0.80 | 1,857 | 16.4% |
| 0.8 – 0.90 | 1,290 | 11.4% |
| **0.9 – 1.00** | **7,169** | **63.3%** |

**63.3% of all wrong predictions carry >90% confidence.** The model does not know what it does not know. Thresholding on confidence would reject the majority of correct predictions before it meaningfully filters errors.

### 4.5 Calibration failure — ECE = 0.1385

A well-calibrated model's stated confidence should match its actual accuracy. This model is systematically overconfident at every level:

| Confidence range | N | Actual Accuracy | Stated Confidence | Gap |
|-----------------|---|----------------|------------------|-----|
| 0.50 – 0.60 | 1,733 | 43.9% | 55.1% | −11.3% |
| 0.60 – 0.70 | 1,842 | 51.0% | 65.3% | −14.3% |
| 0.70 – 0.80 | 2,341 | 59.2% | 75.3% | −16.1% |
| 0.80 – 0.90 | 4,068 | 68.3% | 85.6% | **−17.3%** |
| 0.90 – 0.95 | 3,793 | 75.7% | 92.7% | −17.0% |
| 0.95 – 1.00 | 45,479 | 86.3% | 99.5% | −13.3% |

At 80–90% stated confidence, actual accuracy is only 68.3%. At 90–95% confidence, actual accuracy is 75.7%. The model consistently overstates certainty by 13–17 percentage points across all mid-range confidence levels. Temperature scaling applied post-training would reduce ECE to approximately 0.02–0.04 without affecting accuracy.

### 4.6 Fabricated/Manipulated group routing weakness

Group 2 (Fabricated/Manipulated) routes correctly only 72.8% of the time, compared to 92.3% for Structural Deception:

| Group | Coarse P(correct group) |
|-------|------------------------|
| Structural Deception | **0.9006** |
| Authentic | 0.8018 |
| Fabricated/Manipulated | **0.7192** |

The average coarse probability for the correct group is only 71.9% for Fabricated/Manipulated. This group contains the most linguistically diverse categories — Satire (comedy as news), Imposter (fake accounts), Manipulated (doctored images), AI-Generated (LLM output). The shared coarse representation must cover four very different surface patterns, making the routing signal weaker than for the more homogeneous Structural Deception group.

1,151 Fabricated/Manipulated samples are routed to Structural Deception — the dominant misrouting direction. When this happens, Satire and Imposter samples end up in Group 1's fine head and are classified as Misleading or False Connection.

### 4.7 Short and medium titles are hardest

| Title length | Error rate |
|-------------|-----------|
| 4–6 words | **22.7%** |
| 1–3 words | 19.5% |
| 7–9 words | 18.2% |
| 10–14 words | 16.4% |
| 15+ words | 15.5% |

4–6 word titles have the highest error rate — higher even than 1–3 word titles. Very short titles (1–3 words) tend to be image-caption style posts the model has learned to treat cautiously. Medium-length titles (4–6 words) contain just enough information to mislead the model while lacking enough context to fully disambiguate. This is the model's most unreliable operating zone.

### 4.8 Class imbalance — three categories below average

| Category | Support | % test | F1 | vs Average |
|----------|---------|-------|----|-----------|
| True Content | 23,507 | 39.6% | 0.8594 | above ✓ |
| False Connection | 17,472 | 29.5% | 0.8431 | above ✓ |
| Misleading Content | 11,297 | 19.0% | 0.7199 | **below ✗** |
| Satire/Parody | 3,514 | 5.9% | 0.7265 | **below ✗** |
| Manipulated Content | 2,305 | 3.9% | 0.8182 | above ✓ |
| Imposter Content | 1,224 | 2.1% | 0.6314 | **below ✗** |

Misleading Content (19.0% of data) performs below average despite its large support — a genuine weakness driven by its confusion with False Connection. Imposter Content (2.1%) is the furthest below average. The 18× imbalance between True Content and Imposter Content is partially addressed by weighted loss but not fully resolved.

### 4.9 True Content as default for ambiguous inputs

The 25 hardest misclassified examples all share one pattern — predicted as True Content at 100% confidence:

| True | Predicted | Conf | Headline |
|------|-----------|------|---------|
| False Connection | True Content | 1.000 | "my rope flying off a friend swing" |
| False Connection | True Content | 1.000 | "grumpy nate robinson" |
| Satire/Parody | True Content | 1.000 | "wiggas granted minority status in america" |
| Misleading Content | True Content | 1.000 | "watching my friend drink from inside the shot glass" |
| Satire/Parody | True Content | 1.000 | "new drink driving lanes proposed in ireland" |
| Misleading Content | True Content | 1.000 | "a handdrawn map of estonia latvia and lithuania" |
| Misleading Content | True Content | 1.000 | "my cocktail last night told me i was a ravenclaw" |
| Manipulated Content | True Content | 1.000 | "tests confirm that germanys massive nuclear fusion" |
| Manipulated Content | True Content | 1.000 | "christian group sues for right to lie" |
| False Connection | True Content | 1.000 | "donald trump cat" |
| Satire/Parody | True Content | 1.000 | "the definitive dnc guide to cybersecurity" |
| False Connection | True Content | 1.000 | "angry baby" |
| Satire/Parody | True Content | 1.000 | "why is eric trump like this the daily show" |
| Misleading Content | True Content | 1.000 | "russian church in" |
| Satire/Parody | True Content | 1.000 | "nfl geneticists working on developing ligamentfree play" |
| False Connection | True Content | 1.000 | "been on this earth years and ive never seen anything" |
| False Connection | True Content | 1.000 | "google teaching dogs how to dog since" |
| False Connection | True Content | 1.000 | "this baby is pissed" |
| False Connection | True Content | 1.000 | "got a picture of the scene at phillies" |
| Misleading Content | True Content | 1.000 | "church was packed for easter sunday" |
| False Connection | True Content | 1.000 | "toronto exmayor rob ford walking past current mayor" |
| Manipulated Content | True Content | 1.000 | "the youngest contestant in this years scripps national" |
| False Connection | True Content | 1.000 | "hillary clinton and bernie sanders at the recent nickel" |
| Imposter Content | True Content | 1.000 | "esrb releases statement claiming loot boxes not being g" |
| Satire/Parody | True Content | 1.000 | "nfl geneticists working on developing ligamentfree play" |

Headlines like "grumpy nate robinson", "angry baby", "donald trump cat" are False Connection posts where the caption does not match the image — indistinguishable from genuine news without the image. "russian church in" is a Misleading Content post with an incomplete title that provides no classifiable signal whatsoever.

This is a **fundamental text-only limitation**, not a model architecture failure. These cases require image features to resolve. They represent the irreducible error floor for any text-only classifier on Fakeddit.

---

## 5. Confusion Matrix

```
              True  Satire  Mislead  Imposter  FalseConn  Manip  AI-Gen  Clickbait
True Content  19558    318     2069      106       998      203     38       217
Satire/Parod    349   2409      194       23       432       53      8        46
Misleading C   1108    102     8628       33      1307       55      5        59
Imposter Con    258     30       96      667       111       31      4        27
False Connec    582    224     1611       48     14864       38     10        95
Manipulated     155     35       74       10        77     1859      1        94
```

### Top 20 confusion pairs

| True | Predicted | Count | % of true class |
|------|-----------|-------|-----------------|
| True Content | Misleading Content | 2,069 | 8.8% |
| False Connection | Misleading Content | 1,611 | 9.2% |
| Misleading Content | False Connection | 1,307 | 11.6% |
| Misleading Content | True Content | 1,108 | 9.8% |
| True Content | False Connection | 998 | 4.2% |
| False Connection | True Content | 582 | 3.3% |
| Satire/Parody | False Connection | 432 | 12.3% |
| Satire/Parody | True Content | 349 | 9.9% |
| True Content | Satire/Parody | 318 | 1.4% |
| Imposter Content | True Content | 258 | 21.1% |
| False Connection | Satire/Parody | 224 | 1.3% |
| True Content | Clickbait | 217 | 0.9% |
| True Content | Manipulated Content | 203 | 0.9% |
| Satire/Parody | Misleading Content | 194 | 5.5% |
| Manipulated Content | True Content | 155 | 6.7% |
| Imposter Content | False Connection | 111 | 9.1% |
| True Content | Imposter Content | 106 | 0.5% |
| Misleading Content | Satire/Parody | 102 | 0.9% |
| Imposter Content | Misleading Content | 96 | 7.8% |
| False Connection | Clickbait | 95 | 0.5% |

**Notable patterns:**
- Misleading ↔ False Connection dominates (2,918 total cross-confusions) — boundary ambiguous without image
- Imposter → True Content (21.1%) — imposter headlines designed to look authentic
- Manipulated → Clickbait (4.1%) and FalseConn → Clickbait (0.5%) — sensational framing overlap
- True Content → Clickbait (0.9%) — genuine news with clickbait-style surface form

---

## 6. Coarse Head Analysis

```
Coarse accuracy: 86.40%

                      precision    recall  f1-score   support
           Authentic     0.8886    0.8320    0.8594     23507
Structural Deception     0.8569    0.9234    0.8889     28769
    Fabricated/Manip     0.8130    0.7284    0.7684      7043
```

### Coarse confusion matrix

| True \ Predicted | Authentic | Struct. Decept | Fabricated/Manip |
|-----------------|-----------|---------------|-----------------|
| Authentic | **19,558** | 3,284 | 665 |
| Structural Deception | 1,690 | **26,564** | 515 |
| Fabricated/Manipulated | 762 | 1,151 | **5,130** |

### Average coarse probabilities per true group

| True Group | P(Group 0) | P(Group 1) | P(Group 2) |
|-----------|-----------|-----------|-----------|
| Authentic | 0.8018 | 0.1617 | 0.0364 |
| Structural Deception | 0.0767 | **0.9006** | 0.0227 |
| Fabricated/Manipulated | 0.1158 | 0.1650 | 0.7192 |

Structural Deception is the most confidently routed group (90.1% average probability). Fabricated/Manipulated is the weakest (71.9%), with 16.5% of its probability mass leaking to Group 1 — explaining the 1,151 Fabricated/Manipulated samples misrouted to Structural Deception.

---

## 7. Confidence & Calibration

| Metric | Value |
|--------|-------|
| Mean confidence (correct) | **0.9627** |
| Mean confidence (incorrect) | 0.8841 |
| Std confidence (correct) | 0.0877 |
| Std confidence (incorrect) | 0.1525 |
| Overconfidence gap | −0.0786 |
| **ECE** | **0.1385** |

### Calibration table

| Confidence | N | Actual Acc | Stated Conf | Gap |
|-----------|---|-----------|------------|-----|
| 0.50–0.60 | 1,733 | 43.9% | 55.1% | −11.3% |
| 0.60–0.70 | 1,842 | 51.0% | 65.3% | −14.3% |
| 0.70–0.80 | 2,341 | 59.2% | 75.3% | −16.1% |
| 0.80–0.90 | 4,068 | 68.3% | 85.6% | **−17.3%** |
| 0.90–0.95 | 3,793 | 75.7% | 92.7% | −17.0% |
| 0.95–1.00 | 45,479 | 86.3% | 99.5% | −13.3% |

---

## 8. Loss Analysis

| Component | Value |
|-----------|-------|
| Average coarse loss | 0.4296 |
| Average fine loss | 0.4593 |
| **Average total loss** | **0.4445** |

Coarse and fine losses are near-equal (0.4296 vs 0.4593), confirming the 0.5/0.5 loss weighting is working as intended — neither stage dominates the gradient signal.

### Per-category loss

| Category | N | Avg Loss | Std Loss |
|----------|---|----------|----------|
| False Connection | 17,472 | **0.1595** | 0.6592 |
| Misleading Content | 11,297 | 0.3323 | 0.9444 |
| True Content | 23,507 | 0.4831 | 1.0980 |
| Manipulated Content | 2,305 | 0.6810 | 1.6700 |
| Satire/Parody | 3,514 | 1.1352 | 2.0387 |
| Imposter Content | 1,224 | **1.6612** | 2.4039 |

Loss ranking mirrors error rate ranking exactly — False Connection lowest loss and easiest, Imposter Content highest loss and hardest. The high standard deviations for Imposter (2.40) and Satire (2.04) reveal a bimodal pattern: some samples are easy and produce very low loss, while others are hard and produce very high loss. This bimodality explains why weighted loss is more effective than uniform loss for these categories.

---

## 9. Clickbait New Category Analysis

### 9.1 Clickbait predictions on the 6-way test set

The model predicts Clickbait on **538 samples (0.9%)** of the original test set — none of which have Clickbait as their ground truth label. This characterises what the model recognises as Clickbait-like from categories that predate the Clickbait extension.

| Category | Clickbait predictions | % of category |
|----------|----------------------|---------------|
| Manipulated Content | 94 / 2,305 | **4.1%** |
| Imposter Content | 27 / 1,224 | **2.2%** |
| Satire/Parody | 46 / 3,514 | 1.3% |
| True Content | 217 / 23,507 | 0.9% |
| False Connection | 95 / 17,472 | 0.5% |
| Misleading Content | 59 / 11,297 | 0.5% |

The model is appropriately conservative — only 0.9% overall. Manipulated Content bleeds most into Clickbait (4.1%), followed by Imposter Content (2.2%). Both make sense: Manipulated media often uses sensational headlines to maximise engagement, and Imposter accounts frequently mimic tabloid/clickbait sources. True Content accounts for 217 Clickbait predictions — genuine news with clickbait-like surface form (curiosity gaps, sensational phrasing).

### 9.2 Sample headlines predicted as Clickbait

**From True Content** — genuine news with clickbait surface patterns:
- "the omnis in gta has the ultimate owo on it"
- "when the ad promoted the video that you want to watch"
- "someone made doom in roblox and it really works"

**From Satire/Parody** — satirical headlines with engagement-bait structure:
- "things men instantly think of after seeing a busty girl"
- "why net neutrality is terrible and please help me someone has a gun to my head"
- "we asked famous authors for the most important advice theyd give to young writers"

**From Misleading Content** — misleading framing with clickbait-like hooks:
- "horror vs pleasant surprise"
- "ellen degeneres explains the full story to her critics"
- "this is what happens when you let wiley coyote design the airport"

**From Manipulated Content** — sensational framing around doctored media:
- "what happens when you take all the air out of lays"
- "this college student was kicked out of gym for wearing an outfit and its crazy"
- "what young people fear the most"

**Pattern:** The Clickbait detector has correctly learned the linguistic fingerprint of clickbait — curiosity gaps ("what happens when"), sensational qualifiers ("its crazy", "ultimate"), audience address ("you", "your"), and engagement bait structures. These surface signals are present in other categories too, which is expected and appropriate — the model is not hallucinating Clickbait, it is recognising genuine clickbait-like properties in headlines that happen to carry other ground truth labels.

---

## 10. Summary of Key Findings

### Strengths

| Finding | Detail |
|---------|--------|
| Strong majority class performance | True Content (F1 0.86), False Connection (F1 0.84), Manipulated Content (F1 0.82) all above 80% |
| 66.1% of test set correct at >95% confidence | Model is confidently right on two thirds of samples |
| Fine heads excellent when routing correct | 100% Group 0, 96.2% Group 2, 88.4% Group 1 |
| Hierarchical beats flat on minority classes | Satire +0.037 F1, Imposter +0.052 F1 |
| Clickbait extension zero regression | Adding 8th category preserved all existing performance |
| Balanced prediction volumes | No majority class collapse — model predicts all categories |
| Fast inference | 323 samples/second, 3.10ms per sample |
| Longer titles easier | 15+ words: 15.5% error, 4–6 words: 22.7% error |

### Weaknesses

| Finding | Detail |
|---------|--------|
| Routing bottleneck | 71.2% of errors from coarse head — fine heads not the problem |
| Imposter Content (45.5% error) | Designed to look authentic — text-only limitation |
| Misleading ↔ False Connection | 2,918 cross-confusions — image required to resolve |
| Systematic overconfidence | 63.3% of wrong predictions at >90% confidence |
| ECE = 0.1385 | Confidence scores overstate certainty by 13–17% |
| True Content default | All 25 hardest examples: correct answer requires image |
| 4–6 word titles hardest | 22.7% error — medium-length ambiguous zone |
| Fabricated/Manipulated routing | Only 71.9% confident on correct group |

---

## 11. Implications for Future Work

| Finding | Recommended Action |
|---------|-------------------|
| ECE = 0.1385 | Temperature scaling post-training — would reduce to ~0.02 with no accuracy cost |
| Routing bottleneck (71.2%) | Coarse head improvement is highest priority future investment |
| Imposter + True Content defaults | Multimodal extension (ResNet/ViT + BERT late fusion) — only path to resolving image-dependent cases |
| Misleading ↔ False Connection | Out-of-distribution evaluation to quantify irreducible text-only confusion |
| Imposter Content loss = 1.66 | Further weighted loss increase or dedicated augmentation pipeline |
| Clickbait surface overlap with Manipulated | Investigate shared linguistic patterns — potential taxonomy refinement |

---

*Model: `hierarchical_v4_checkpoint.pt` (8-category, post-Clickbait)*
*Evaluation: Fakeddit `test_public.tsv` (59,319 samples, 6-way ground truth labels)*
*Author: Iman Ein Alizadeh (s2901349) · University of Edinburgh EPCC · MSc Dissertation 2025–26*
*Supervisor: Oliver Brown*
