# Projects

Four independent engineering projects spanning high-performance computing, computer vision, natural language processing, and embedded robotics. Each is a complete, self-contained repository — real source code, automated tests, reproducible results, and documentation covering not just what was built but the design decisions and debugging history behind it.

![C](https://img.shields.io/badge/C-00599C?style=flat&logo=c&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Java](https://img.shields.io/badge/Java-ED8B00?style=flat&logo=openjdk&logoColor=white)
![MPI](https://img.shields.io/badge/MPI-Parallel_Computing-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white)

| | Domain | Language | Tests | Verified on |
|---|---|---|---|---|
| [Parallel Sandpile HPC](parallel-sandpile-hpc/) | HPC / Parallel Computing | C + MPI | 7 passing | ARCHER2 supercomputer |
| [Exercise Biomechanics CV](exercise-biomechanics-cv/) | Computer Vision | Python | 15 passing | MediaPipe 0.10.14 |
| [Research Vault](research-vault/) | NLP / Data Systems | Python | 7 passing | Flask + SQLite |
| [Swift Bot Control](swift-bot-facial-expression-control/) | Robotics / CV | Python + Java | 7 passing | Live TCP integration |

---

## 1. Parallel Abelian Sandpile — MPI & High-Performance Computing
**[`parallel-sandpile-hpc/`](parallel-sandpile-hpc/)**

A 2D-decomposed MPI implementation of the Bak–Tang–Wiesenfeld self-organised-criticality sandpile model, built for MSc coursework and benchmarked at scale on the **ARCHER2 national supercomputer**.

**Measured strong scaling on ARCHER2:**

| Ranks | Nodes | Wall time | Speedup | Efficiency |
|---:|---:|---:|---:|---:|
| 4 | 1 | 177.4s | 1.00× | 100% |
| 16 | 1 | 66.9s | 2.65× | 66.3% |
| **32** | **2** | **36.8s** | **4.82×** | **60.2%** |

**Engineering highlights:**
- 2D Cartesian process topology (`MPI_Dims_create` + `MPI_Cart_create`) with mixed boundary conditions — non-periodic top/bottom, periodic left/right
- Non-blocking halo exchange (`MPI_Isend`/`MPI_Irecv` posted before any `MPI_Waitall`) so the four neighbour directions overlap instead of serialising
- **Correctness-by-construction:** the initial condition is a deterministic hash of each cell's *global* coordinate rather than a per-rank seed, guaranteeing the converged result is provably identical whether run on 1 rank or 32 — enforced by an automated test, not eyeballed
- Full design rationale and a documented performance bug (allocation inside the per-step loop, fixed) in `docs/design.md`

**Stack:** C · MPI · SLURM · Python (analysis/plotting) · pytest

---

## 2. Exercise Biomechanics Analyser — Real-Time Computer Vision
**[`exercise-biomechanics-cv/`](exercise-biomechanics-cv/)**

Real-time squat form analysis from a standard webcam — no wearables, no sensors. Tracks 33 MediaPipe pose landmarks, computes joint angles, and counts reps through a **noise-robust threshold-crossing state machine** rather than naive local-minima detection (which over-counts on real per-frame jitter — see the regression test that catches it).

**Two generations, one repo:**
- **Squat Analyser** — live angle tracking, rep counting, depth scoring, full session export
- **Ghost Trainer** — an AR overlay showing a phase-synced "ideal form" reference (rule-based or extracted from any recorded video) so form deviation is visible *as it happens*, colour-coded green/amber/red by how far the live pose has drifted

**Engineering highlights:**
- Phase-percentage indexing (not wall-clock time) keeps the reference skeleton synchronised regardless of how fast the lifter moves
- 15 automated tests covering the angle geometry and rep-counting state machine, including the noise-robustness regression test
- Jupyter notebook that runs end-to-end and produces real trajectory/depth-quality plots from an exported session

**Stack:** Python · MediaPipe · OpenCV · NumPy · Matplotlib · pytest

---

## 3. Research Vault — Academic Paper Management System
**[`research-vault/`](research-vault/)**

A group project (**Ctrl-Alt-Elite**, EPCC MSc Practical Software Development) for ingesting, categorising, and cross-referencing research papers at scale. Parses PDFs, classifies by keyword rules, and computes **two independent similarity engines** — heuristic keyword-overlap and semantic spaCy-vector cosine similarity — plus named-entity co-occurrence network analysis.

**Engineering highlights:**
- Ships as both an interactive CLI and a Flask web app sharing the same SQLite backend and business logic
- **Real bug found and fixed:** the heuristic similarity engine's common-word filter broke on small corpora (any word appearing in even 1 of 4 papers already exceeds the 12% threshold, wrongly discarding it as "too common") — fixed with an absolute-count fallback, verified by a regression test
- Known model limitation surfaced rather than hidden: `en_core_web_sm` has no word vectors, so semantic scores silently come out as 0.0 without the larger model — now raises an explicit warning instead of failing silently

**Stack:** Python · Flask · SQLite · spaCy · PyMuPDF · NetworkX · pytest

---

## 4. Adaptive Swift Bot Control — Facial Expression Recognition & Robotics
**[`swift-bot-facial-expression-control/`](swift-bot-facial-expression-control/)**

A full vision-to-robotics pipeline: a Python client detects the operator's face, classifies their expression, and tracks gaze direction, then drives a **Java-based robot control server** over a live TCP socket — happiness advances, anger retreats, everything else holds position.

**Engineering highlights:**
- Clean separation of concerns: `FaceDetector` → `EmotionDetector` → `GazeTracker` → command mapping → `RobotClient` (Python) → `RobotServer`/`CommandHandler`/`MotorController` (Java)
- Confidence-gated command dispatch — low-confidence emotion guesses default to `HOLD` rather than triggering unwanted movement
- Documented wire protocol (`docs/protocol.md`) with defensive parsing on both ends — malformed commands are logged and dropped, not fatal
- Verified with a genuine end-to-end integration test: real compiled Java server, real socket connection, real Python client, confirmed via server-side command log

**Stack:** Python · Java · OpenCV · MediaPipe · TCP Sockets · pytest

---

## Dissertation

The MSc dissertation project — **Hierarchical BERT Classification for Fine-Grained Misinformation Detection** — lives separately in [`hierarchical-bert-misinformation/`](hierarchical-bert-misinformation/), and is not counted among the four projects above.
