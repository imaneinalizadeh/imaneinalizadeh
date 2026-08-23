# Contributing

Thank you for your interest in this project. This is an academic dissertation
repository for an MSc in High Performance Computing at the University of Edinburgh
(EPCC), supervised by Oliver Brown. Contributions are welcome in the spirit of
open academic collaboration.

---

## Who Can Contribute

- **Researchers** interested in multimodal fake news detection or hierarchical NLP classifiers
- **Students** working on related problems who want to build on this taxonomy or architecture
- **Anyone** who spots a bug, a documentation gap, or a reproducibility issue

---

## What We Welcome

- Bug reports and fixes in training or evaluation scripts
- Improvements to documentation and inline code comments
- Additional experiments using the same Fakeddit-based taxonomy
- Performance optimisations (batching, inference speed, memory usage)
- New category definitions or label mapping proposals with justification

---

## What We Do Not Accept

- Changes that alter the core 8-category taxonomy without academic justification
- Modifications to training data splits (reproducibility must be preserved)
- Code that introduces external dependencies without discussion
- Pull requests without a clear description of what changed and why

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://git.ecdf.ed.ac.uk/EPCC-MSc-Projects/full-time/2025-26/s2901349.git
cd s2901349
```

### 2. Set up your environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Core dependencies:
- `torch >= 2.0`
- `transformers >= 4.46`
- `datasets`
- `scikit-learn`
- `pandas`, `numpy`

### 3. Reproduce a baseline

```bash
# Train the flat baseline
python src/train_flat.py

# Train the hierarchical model
python src/train_hierarchical.py

# Evaluate
python src/evaluate.py --model models/hierarchical_v2_best.pt
```

---

## Project Structure

```
s2901349/
├── data/               # Raw and processed Fakeddit splits
├── models/             # Saved model checkpoints (.pt files)
├── notebooks/          # Experiment notebooks (one per experiment)
├── results/            # Evaluation outputs, confusion matrices, logs
├── src/                # Source code — training, evaluation, data utils
├── CHANGELOG.md        # Version history
├── CONTRIBUTING.md     # This file
├── LICENSE             # MIT License
└── main_README.md      # Project overview
```

---

## Submitting a Merge Request

1. **Fork** the repository or create a feature branch:
   ```bash
   git checkout -b feature/your-descriptive-name
   ```

2. **Make your changes** — keep commits small and focused. Each commit should do
   one thing and have a clear message:
   ```
   fix: correct class weight calculation for imbalanced categories
   feat: add per-epoch F1 logging to training loop
   docs: clarify LOCAL_TO_FINE mapping in README
   ```
   We loosely follow [Conventional Commits](https://www.conventionalcommits.org/).

3. **Test your changes** — at minimum, run the evaluation script and confirm
   metrics match the reported baselines within floating-point tolerance.

4. **Open a Merge Request** on GitLab with:
   - A clear title
   - What the change does and why
   - Any relevant metric changes (before / after)
   - Reference to any related issue

---

## Reporting Issues

Open an issue on GitLab with:
- A short, descriptive title
- Steps to reproduce the problem
- Expected vs actual behaviour
- Your environment (Python version, OS, GPU/CPU)

---

## Code Style

- **Python**: follow [PEP 8](https://peps.python.org/pep-0008/). Max line length 100.
- **Docstrings**: NumPy style for functions with non-obvious behaviour.
- **Notebooks**: clear markdown cells separating each experiment section.
  Output cells should be cleared before committing.
- **No magic numbers**: constants should be named and defined at the top of the file.

---

## Academic Integrity

This project is submitted as part of an MSc dissertation. Any contribution that
could constitute plagiarism or misrepresentation of original work is strictly
prohibited. If you are building on this work in your own academic submission,
please cite appropriately:

```
Iman Ein Alizadeh. "Hierarchical Multimodal Deep Learning for Fake News Detection."
MSc Dissertation, University of Edinburgh — EPCC, 2026.
```

---

## Contact

**Iman Ein Alizadeh**
MSc High Performance Computing, University of Edinburgh
Student ID: s2901349
Supervisor: Oliver Brown (EPCC)
