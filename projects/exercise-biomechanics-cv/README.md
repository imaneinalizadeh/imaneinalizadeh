# Exercise Biomechanics CV

Real-time squat form analysis using MediaPipe Pose — starting from a
rep counter and depth scorer, extended into an AR "ghost trainer" that
overlays correct-form guidance on the live feed as you move.

## Two generations, in this repo

### 1. Squat Form Analyser (`src/squat_analyzer.py`)

Tracks knee/hip/torso angles from 33 MediaPipe pose landmarks, counts
reps, scores squat depth against a parallel threshold, and exports a
full per-frame session log plus a summary (reps, success rate, best/
average depth).

```bash
python3 src/squat_analyzer.py --video my_session.mp4 --export results/session.json
```

### 2. Ghost Trainer (`src/ghost_trainer.py`, `ghost_mirror.py`, `extract_ghost_from_video.py`)

Instead of scoring form only after a rep finishes, this overlays a
translucent "ghost" showing what correct form looks like **at the
current instant**, colour-coded by how far the live pose has drifted
(green/amber/red). Two flavours:

- **Rule-based ghost** (`ghost_trainer.py`) — a synthetic cosine-
  interpolated reference, no recorded video needed.
- **Real-video ghost** (`ghost_mirror.py` + `extract_ghost_from_video.py`)
  — extracts the cleanest rep from any downloaded tutorial clip or
  your own "textbook" rep, and plays it back split-screen, phase-
  matched to your live feed.

```bash
# Build a reference from a video of correct form
python3 src/extract_ghost_from_video.py --video textbook_squat.mp4 \
    --out data/ghost_reference/ghost_table_squat.json

# Live overlay against the rule-based ghost
python3 src/ghost_trainer.py --video my_attempt.mp4

# Split-screen against a real recorded reference
python3 src/ghost_mirror.py --live my_attempt.mp4 --reference textbook_squat.mp4 \
    --ghost-table data/ghost_reference/ghost_table_squat.json
```

**The hard problem this solves: phase alignment.** Two people (or the
same person on different reps) never move at the same speed, so
indexing the ghost by wall-clock time drifts out of sync within a
second. Every script here indexes the ghost by **phase percentage**
(0% = standing, 100% = deepest point of the rep) computed from the
live knee angle itself — see [`docs/architecture.md`](docs/architecture.md).

## Why the rep counter uses threshold-crossing, not local minima

An earlier version flagged a rep whenever the knee-angle signal hit a
local minimum. On real webcam data, MediaPipe's per-frame jitter
creates dozens of spurious tiny local minima even while someone holds
still at the bottom of a rep — this over-counted badly. The fix,
in [`src/rep_counter.py`](src/rep_counter.py): a rep only counts once
the signal crosses from above a *standing* threshold to below a
*bottom* threshold and back — single-frame noise can't cross a
threshold the way it can create a local minimum. See
`tests/test_rep_counter.py::test_counts_correctly_with_realistic_noise`
for the regression test.

## Repository structure

```
exercise-biomechanics-cv/
├── src/
│   ├── angle_calculator.py       Pure geometry (no MediaPipe dependency — easy to unit test)
│   ├── pose_utils.py             MediaPipe Pose wrapper
│   ├── rep_counter.py            Threshold-crossing rep/phase state machine
│   ├── squat_analyzer.py         Main analyser (generation 1)
│   ├── ghost_reference.py        Ghost table build + phase-interpolated lookup
│   ├── ghost_trainer.py          Rule-based ghost overlay (generation 2)
│   ├── ghost_mirror.py           Real-video split-screen ghost (generation 2)
│   └── extract_ghost_from_video.py   Builds a ghost table from any reference video
├── data/
│   ├── sample_session/pose_session_example.json   Synthetic demo session (see notebook)
│   └── ghost_reference/ghost_table_squat.json     Generated rule-based ghost table
├── notebooks/
│   └── analysis.ipynb            Loads a session JSON, plots trajectory + depth-per-rep
├── results/
│   ├── knee_angle_trajectory.png
│   ├── depth_per_rep.png
│   └── session_summary_example.md
├── tests/
│   ├── test_angle_calculator.py
│   └── test_rep_counter.py
└── docs/
    └── architecture.md
```

## Installation

```bash
pip install -r requirements.txt
```

**Important:** `mediapipe` is pinned to `0.10.14`. MediaPipe 1.0.0+
removed the legacy `mp.solutions` API entirely
(`AttributeError: module 'mediapipe' has no attribute 'solutions'`),
which `pose_utils.py` depends on.

## Running the tests

```bash
python3 -m pytest tests/ -v
```

15 tests covering the angle geometry and the rep-counter state
machine, including the noisy-signal regression test above. These run
without a camera or MediaPipe model download — they exercise the pure
Python logic directly.

## Author

Iman Ein Alizadeh.
