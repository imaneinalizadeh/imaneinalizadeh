# Projects

Four independent projects spanning high-performance computing, computer vision, natural language processing, and embedded/robotics software — each a complete repository with source code, tests, documentation, and reproducible results.

---

## 1. Parallel Abelian Sandpile — MPI & High-Performance Computing
**[`parallel-sandpile-hpc/`](parallel-sandpile-hpc/)**

A 2D-decomposed MPI implementation of the Bak-Tang-Wiesenfeld sandpile cellular automaton, built for MSc coursework and run at scale on the **ARCHER2** national supercomputer. Uses non-blocking halo exchange (`MPI_Isend`/`MPI_Irecv`) across a Cartesian process topology, achieving **4.82× speedup on 32 ranks across 2 nodes**. Includes an automated correctness suite verifying the result is identical regardless of how many processes it runs on.

**Stack:** C, MPI, SLURM, Python (analysis/plotting)

---

## 2. Exercise Biomechanics Analyser — Real-Time Computer Vision
**[`exercise-biomechanics-cv/`](exercise-biomechanics-cv/)**

Real-time squat form analysis from a standard webcam — no wearables or sensors. Tracks joint angles via MediaPipe pose estimation, counts reps with a noise-robust threshold-crossing state machine, and scores depth against a parallel-squat threshold. Extended into an AR "ghost trainer" that overlays a phase-synced reference skeleton on the live feed, so form deviation is visible as it happens rather than scored after the fact.

**Stack:** Python, MediaPipe, OpenCV, NumPy, Matplotlib

---

## 3. Research Vault — Academic Paper Management System
**[`research-vault/`](research-vault/)**

A group project (Ctrl-Alt-Elite, EPCC MSc Practical Software Development) for ingesting, categorising, and cross-referencing research papers. Parses PDFs, classifies them by keyword rules, and computes both heuristic (keyword-overlap) and semantic (spaCy vector) similarity between papers, plus named-entity co-occurrence analysis. Ships as both a CLI and a Flask web interface sharing the same SQLite backend.

**Stack:** Python, Flask, SQLite, spaCy, PyMuPDF, NetworkX

---

## 4. Adaptive Swift Bot Control — Facial Expression Recognition & Robotics
**[`swift-bot-facial-expression-control/`](swift-bot-facial-expression-control/)**

A vision-to-robotics pipeline: a Python client detects the operator's face, classifies their expression, and tracks gaze direction, then sends movement commands over a TCP socket to a Java-based robot control server. Happiness and anger map to forward/backward movement, with a confidence threshold preventing the robot from reacting to low-confidence guesses.

**Stack:** Python, Java, OpenCV, MediaPipe, TCP Sockets

---

## Dissertation

The MSc dissertation project — **Hierarchical BERT Classification for Fine-Grained Misinformation Detection** — lives separately in [`hierarchical-bert-misinformation/`](hierarchical-bert-misinformation/).
