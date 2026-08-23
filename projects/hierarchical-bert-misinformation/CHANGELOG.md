# Changelog

All notable changes to this project are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/).

---

## [Unreleased]

No remaining experimental work. All experiments complete. Report submitted 17 April 2026. Presentation delivered 20 May 2026.

---

## [1.3.0] — 2026-07-06 — Repository Reorganisation

### Changed
- `data/` reorganised into subfolders:
  - `data/fakeddit/` — train.tsv, validate.tsv, test_public.tsv
  - `data/ai_generated/` — AI-Generated Content experiment data and builder scripts
  - `data/clickbait/` — Clickbait extension dataset
  - `data/extensibility/` — Conspiracy Theory, Junk News, Propaganda datasets
- `models/` reorganised into subfolders:
  - `models/baseline/` — best_model.pt, flat_best_model.pt
  - `models/ai_generated/` — hierarchical_v2_best.pt, hierarchical_v2_headlines_best.pt, hierarchical_v3_best.pt
  - `models/clickbait/` — hierarchical_v4_best.pt, hierarchical_v4_checkpoint.pt
  - `models/production/` — hierarchical_v5_best.pt (recommended)
  - `models/extensibility/` — hierarchical_v6_junknews.pt, hierarchical_v6_propaganda.pt
- `results/` reorganised into subfolders:
  - `results/experiment1/` — hierarchical vs flat
  - `results/experiment2/` — AI-Generated Content (3 attempts)
  - `results/experiment3/` — Clickbait extension
  - `results/experiment4/` — taxonomy extensibility analysis
  - `results/error_analysis/` — post-Clickbait error analysis
- `src/` reorganised into subfolders:
  - `src/training/baseline/` — train_hierarchical.py, train_flat_baseline.py
  - `src/training/ai_generated/` — add_ai_attempt1/2/3_essays.py
  - `src/training/retraining/` — train_v5_resume.py
  - `src/training/extensibility/` — round1_conspiracy_train.py, round2_propaganda_train.py
  - `src/data_builders/` — clickbait_dataset_builder.py, round1_conspiracy_dataset.py
  - `src/analysis/` — generate_error_analysis.py
- All READMEs updated to reflect new paths — `README.md`, `src/src_README.md`, `models/models_README.md`
- `results/experiment4/extensibility_analysis.md` updated with full threshold justification section — explains why each criterion threshold was chosen (0.65, 0.03, +5%, 0.20) and what would happen with stricter or looser values
- `README.md` restructured around three research questions — architecture diagram and freezing explanation added to RQ1

---

## [1.2.0] — 2026-06-25 — Taxonomy Extensibility Analysis

### Added
- `results/experiment4/extensibility_analysis.md` — full extensibility experiment answering: "When does adding a new category stop being beneficial?"
- `src/data_builders/round1_conspiracy_dataset.py` — Conspiracy Theory data search script (abandoned, 14 samples found)
- `src/training/extensibility/round1_conspiracy_train.py` — Conspiracy Theory training script (not run — dataset too small)
- `src/training/extensibility/round2_propaganda_train.py` — Propaganda extension attempt (trained, failed viability)
- `data/extensibility/conspiracy_dataset.json` — 14 samples
- `data/extensibility/junknews_dataset.json` — 10,736 samples
- `data/extensibility/propaganda_dataset.json` — 13,456 samples
- `models/extensibility/hierarchical_v6_junknews.pt` — Junk News attempt checkpoint (not viable)
- `models/extensibility/hierarchical_v6_propaganda.pt` — Propaganda attempt checkpoint (not viable)

### Experiment Design

Four strict viability criteria established before any experiments were run — a category addition is only viable if ALL pass:

| # | Criterion | Threshold |
|---|-----------|-----------|
| 1 | New category F1 | > 0.65 |
| 2 | Max existing category F1 drop | < 0.03 |
| 3 | Routing error rate change | < +5% |
| 4 | Expected Calibration Error | < 0.20 |

### Results — 4 Category Addition Attempts

| Attempt | Category | Group | Samples | Result |
|---------|----------|-------|---------|--------|
| 1 | Conspiracy Theory | 2 | 14 | Abandoned — insufficient data |
| 2 | Rumour | 1 | 1,419 (noisy) | Abandoned — insufficient/noisy data |
| 3 | Junk News | 1 | 10,736 | **FAILED — Criterion 1** (F1 0.155) |
| 4 | Propaganda | 2 | 13,456 | **FAILED — Criteria 1 & 3** (F1 0.133, routing +7.6%) |

### Key Findings
- Two independent trained attempts (different groups, different data sources) both failed to reach F1 > 0.65
- Neither caused meaningful regression on existing categories (Criterion 2 passed both times)
- Calibration remained excellent throughout (Criterion 4 passed both times)
- Junk News failed on intra-group discriminability — overlap with Misleading/Clickbait
- Propaganda failed more severely — discriminability AND routing — due to thin text signal for fundamentally visual content
- **Conclusion:** the 8-category taxonomy is at or near the practical extensibility limit for text-only hierarchical BERT on Fakeddit-style content

---

## [1.1.0] — 2026-06-11 — V5 Model Retraining (Error Analysis Fixes)

### Added
- `src/training/retraining/train_v5_resume.py` — retraining script applying all fixes identified from post-Clickbait error analysis

### Improved
- `models/production/hierarchical_v5_best.pt` — retrained from `hierarchical_v4_checkpoint.pt` with four targeted fixes:
  - **Weighted cross-entropy:** Imposter 5×, Satire 3×, Manipulated 3×, AI-Gen 2× — targets categories with highest loss and error rate
  - **Differential learning rates:** BERT 1e-5, coarse head 3e-5, fine heads 2e-5 — coarse head gets highest LR to attack the routing bottleneck (71.2% of errors)
  - **Label smoothing 0.1** — penalises overconfidence during training, targets ECE 0.1385
  - **Partial BERT unfreeze** — allows encoder to adapt to hard cases, targets Fabricated/Manipulated group routing weakness (71.9% coarse confidence)

### Results (1 epoch completed)

| Model | Accuracy | Macro F1 | Notes |
|-------|----------|----------|-------|
| `hierarchical_v4_checkpoint.pt` | 80.89% | 0.7665 | Baseline — post-Clickbait |
| `hierarchical_v5_best.pt` | **82.43%** | **0.7755** | After 1 training epoch |

### Per-category F1 improvement

| Category | v4 F1 | v5 F1 | Delta |
|----------|-------|-------|-------|
| True Content | 0.8594 | 0.868 | +0.009 |
| Satire/Parody | 0.7265 | 0.730 | +0.004 |
| Misleading Content | 0.7199 | 0.733 | +0.013 |
| Imposter Content | 0.6319 | 0.640 | +0.008 |
| False Connection | 0.8431 | 0.850 | +0.007 |
| Manipulated Content | 0.8182 | 0.831 | +0.013 |

Every category improved.

---

## [1.0.0] — 2026-04-26 — Full Error Analysis & Repository Finalisation

### Added
- `src/analysis/generate_error_analysis.py` — automated script that runs full inference on the final 8-category model and produces a complete markdown error analysis report
- `results/error_analysis/Error_Analysis.md` — complete post-Clickbait error analysis covering 15 sections

### Key findings from error analysis
- Overall accuracy 80.89% · Weighted F1 0.8139
- 71.2% of all errors are routing errors (coarse head bottleneck) — fine heads are the strength
- Fine head accuracy given correct routing: Group 0 = 100%, Group 2 = 96.2%, Group 1 = 88.4%
- 66.1% of test set classified correctly at >95% confidence
- Imposter Content hardest: 45.5% error rate, avg loss 1.66
- ECE = 0.1385 — systematic overconfidence across all confidence levels
- Adding Clickbait caused zero regression: total errors changed by only 16 out of 59,319

---

## [0.9.0] — 2026-04-17 — Dissertation Submission

### Added
- Final dissertation report submitted to University of Edinburgh
- Presentation slides (10 slides) added to `submission/`
- HuggingFace Spaces demo live: https://huggingface.co/spaces/imaniuoboubouv/fake-news-detector
  - 338,453 browsable real Fakeddit examples
  - Real-time BERT inference
  - Keyword search with relevance sorting and pagination
  - Category browse mode
  - Light/dark mode toggle with swirl particle background

---

## [0.8.0] — 2026-04-10 — Improved V4 Model & Results Documentation

### Added
- `results/experiment2/ai_generated_attempt2.md` — full analysis of AI-Generated Attempt 2 (Reuters headlines, F1 0.71)
- `results/experiment2/ai_generated_attempt3.md` — full analysis of AI-Generated Attempt 3 (NYT titles, F1 0.90)

### Improved
- V4 model retrained with weighted loss and differential learning rates — `models/clickbait/hierarchical_v4_best.pt`
  - Imposter Content weight 5×, Manipulated 4×, Satire 3×, AI-Generated 2×
  - BERT lr=1e-5, classification heads lr=2e-5
  - Accuracy 84%, Macro F1 0.81

---

## [0.7.0] — 2026-03-28 — Clickbait Extension

### Added
- New 8th category: **Clickbait (label 7)** — placed in Group 1 (Structural Deception)
- Group 1 fine head expanded from 2 outputs to 3 (Misleading, False Connection, Clickbait)
- `src/data_builders/clickbait_dataset_builder.py` — builds Clickbait training subset using subreddit membership and 10 linguistic pattern regexes
- `results/experiment3/clickbait_extension.md`

### Results

| Metric | Value |
|--------|-------|
| Clickbait F1 | 0.83 |
| Overall Accuracy | ~80.89% |
| Zero regression | All 6 original categories preserved |

### Notes
- Selective freezing: only Group 1 fine head and coarse head retrained — BERT and Group 2 frozen
- Model saved as `models/clickbait/hierarchical_v4_checkpoint.pt`
- Clickbait succeeded in one attempt — in-domain data, no length/style mismatch

---

## [0.6.0] — 2026-03-20 — AI-Generated Content Detection (Attempt 3)

### Added
- `src/training/ai_generated/add_ai_attempt3_essays.py` — NYT title dataset (`gsingh1-py/train`), 5 LLM sources, ~25,000 samples
- Differential learning rates: BERT lr=1e-5, heads lr=2e-5
- Weighted cross-entropy: AI-Generated 2×, Imposter 3×, Satire 2×

### Results

| Metric | Value |
|--------|-------|
| AI-Generated F1 | **0.90** |
| AI-Generated Precision | 0.92 |
| AI-Generated Recall | 0.86 |
| Overall Accuracy | 84% |
| Macro F1 | 0.81 |

### Notes
- Domain alignment is the key insight: NYT titles match Fakeddit title format perfectly
- F1 0.90 is +0.79 improvement over Attempt 1 and +0.19 over Attempt 2
- Model saved as `models/ai_generated/hierarchical_v3_best.pt`

---

## [0.5.0] — 2026-03-10 — AI-Generated Content Detection (Attempt 2)

### Added
- `src/training/ai_generated/add_ai_attempt2_essays.py` — Reuters headline dataset (`artnitolog/llm-generated-texts`), ~22,000 samples

### Results

| Metric | Value |
|--------|-------|
| AI-Generated F1 | 0.71 |
| AI-Generated Precision | 0.78 |
| AI-Generated Recall | 0.61 |

### Notes
- +0.60 F1 improvement over Attempt 1 — confirms domain mismatch hypothesis
- Reuters still more formal than Reddit titles — motivates Attempt 3
- Model saved as `models/ai_generated/hierarchical_v2_headlines_best.pt`

---

## [0.4.0] — 2026-03-01 — AI-Generated Content Detection (Attempt 1)

### Added
- New 7th category: **AI-Generated Content (label 6)** — placed in Group 2 (Fabricated/Manipulated)
- `src/training/ai_generated/add_ai_attempt1_essays.py` — essay dataset (`artem9k/ai-text-detection-pile`), 30,000 samples
- `results/experiment2/ai_generated_attempt1.md` — intentionally documents failure case

### Results

| Metric | Value |
|--------|-------|
| AI-Generated F1 | 0.11 |
| AI-Generated Precision | 0.94 |
| AI-Generated Recall | 0.18 |

### Notes
- High precision, near-zero recall — signature of domain mismatch
- Essays average 400+ words; Fakeddit titles average 8–12 words
- Key finding: domain match between external data and target distribution is critical
- Model saved as `models/ai_generated/hierarchical_v2_best.pt`

---

## [0.3.0] — 2026-02-28 — Hierarchical vs Flat Classifier Comparison

### Added
- `src/training/baseline/train_hierarchical.py` — full two-stage hierarchical BERT classifier
- `src/training/baseline/train_flat_baseline.py` — flat 6-class BERT baseline for controlled comparison
- `results/experiment1/hierarchical_vs_flat.md`

### Results

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Flat baseline | 81% | 0.75 |
| Hierarchical | 82% | **0.77** |

Statistical reliability: 3 random seeds · p < 0.05 (paired t-test)

### Notes
- Hierarchy benefits minority classes most: Satire +0.04 F1, Imposter +0.04 F1
- 74% of all errors are coarse routing errors — identified as primary bottleneck
- Models saved as `models/baseline/best_model.pt` and `models/baseline/flat_best_model.pt`

---

## [0.2.0] — 2026-02-07 — Data Pipeline & Preprocessing

### Added
- Fakeddit dataset ingestion pipeline (`data/fakeddit/`)
- Label mapping: 6-class Fakeddit → 8-category extensible taxonomy
- BERT tokenisation with `max_length=64`
- `DataLoader` with batch size 64, pin memory, 2 workers
- `notebooks/Fakeddit_overview_.ipynb`

---

## [0.1.0] — 2026-01-22 — Project Initialisation

### Added
- Repository created under EPCC MSc Projects GitLab (s2901349)
- Project structure: `data/`, `models/`, `notebooks/`, `results/`, `src/`, `submission/`
- `.gitignore`, `CONTRIBUTING.md`, `LICENSE`, `CHANGELOG.md`
- Ethics approval submitted to University of Edinburgh Informatics Ethics Committee
- Supervisor: Oliver Brown (EPCC)