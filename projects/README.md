# Projects

Four projects spanning HPC, computer vision, ML, and software engineering. Each has its own folder with full source code, documentation, and instructions.

---

## 1. Parallel Abelian Sandpile — MPI + HPC
**[`parallel-sandpile-hpc/`](parallel-sandpile-hpc/)**

Parallel cellular automaton simulation using MPI 2D domain decomposition and non-blocking communication. Run at scale on the ARCHER2 national supercomputer, achieving 4.82× speedup across 32 cores on 2 nodes.

**Stack:** C, MPI, SLURM, Python (analysis), ImageMagick

---

## 2. Exercise Biomechanics Analyser — Real-Time CV
**[`exercise-biomechanics-cv/`](exercise-biomechanics-cv/)**

Real-time exercise form analysis from a standard webcam. Tracks joint angles for squats, bicep curls, and jump height using MediaPipe pose estimation and OpenCV. No wearables or sensors required.

**Stack:** Python, MediaPipe, OpenCV, NumPy, Matplotlib

---

## 3. Research Vault — Academic Paper Management
**[`research-vault/`](research-vault/)**

Full-stack paper ingestion and categorisation system. Parses PDFs, extracts keywords using spaCy NLP, classifies papers into research areas, and serves a Flask web interface with semantic similarity search.

**Stack:** Python, Flask, SQLite, spaCy, PyMuPDF

---

## 4. MLP Regularisation — Neural Network Overfitting Study
**[`mlp-regularisation/`](mlp-regularisation/)**

From-scratch implementation of Dropout, L1, L2, and Elastic Net regularisation for multi-layer perceptrons in NumPy. Systematic experiments demonstrate the effect on overfitting across network widths and depths.

**Stack:** Python, NumPy, Matplotlib

---

## Dissertation

The MSc dissertation project (Hierarchical BERT Classification for Fine-Grained Misinformation Detection) is in [`hierarchical-bert-misinformation/`](hierarchical-bert-misinformation/).
