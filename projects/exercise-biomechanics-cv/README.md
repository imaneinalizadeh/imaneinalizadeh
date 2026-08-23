# Exercise Biomechanics Analyser — Real-Time Computer Vision

A real-time exercise form analysis system using MediaPipe pose estimation and OpenCV. Tracks joint angles, counts repetitions, measures jump height, and gives live feedback — all from a standard webcam with no wearables or sensors required.

---

## What It Does

Three independent analysers, each targeting a specific exercise:

| Analyser | What It Tracks | Key Metric |
|----------|---------------|------------|
| Squat Analyser | Knee angle at depth | Pass/fail at 90° threshold |
| Bicep Curl Analyser | Elbow flexion angle | Full range of motion check |
| Jump Analyser | Hang time + peak height | Each jump tracked individually |

---

## Architecture

```
Webcam Feed
    │
    ▼
MediaPipe Pose Estimation
(33 landmark keypoints)
    │
    ▼
Joint Angle Calculator
(hip, knee, ankle / shoulder, elbow, wrist)
    │
    ▼
Rep Counter + Form Evaluator
    │
    ▼
Real-Time Overlay (OpenCV)
(angles, rep count, bar chart, feedback text)
```

---

## Demo

### Squat Analyser
- Draws skeleton overlay on live video
- Shows knee angle in real time
- Turns green at correct depth (≤ 90°), red if insufficient
- Counts completed reps automatically

### Jump Analyser
- Calibrates standing height in 2 seconds
- Measures hang time from hip landmark trajectory
- Calculates jump height from hang time: `h = 0.5 × g × (t/2)²`
- Shows last 5 jumps as a bar chart with colour coding
- Tracks personal best

---

## Repository Structure

```
exercise-biomechanics-cv/
├── src/
│   ├── squat_analyser.py          # Squat form analysis
│   ├── bicep_curl_analyser.py     # Bicep curl tracking
│   ├── jump_analyser.py           # Jump height measurement
│   └── angle_utils.py             # Joint angle calculations
├── demos/
│   └── demo_screenshots/          # Example output images
├── docs/
│   └── biomechanics_notes.md      # Joint angle reference guide
├── requirements.txt
└── README.md
```

---

## Getting Started

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run an analyser

```bash
# Squat form
python src/squat_analyser.py

# Bicep curl
python src/bicep_curl_analyser.py

# Jump height
python src/jump_analyser.py
```

Press `q` to quit any analyser.

---

## Dependencies

```
mediapipe>=0.10.0
opencv-python>=4.8.0
numpy>=1.24.0
matplotlib>=3.7.0
```

---

## How Joint Angles Are Calculated

Three landmark points define each angle. For the knee:

```
hip → knee → ankle

angle = arccos( dot(v1, v2) / (|v1| × |v2|) )
```

where `v1 = hip - knee` and `v2 = ankle - knee`.

---

## Jump Height Formula

Hang time `t` is measured from when both feet leave the ground to when they land:

```
height = 0.5 × g × (t / 2)²
```

where `g = 9.81 m/s²`. This gives the centre-of-mass displacement in metres.

---

## Academic Context

Developed as part of a computer vision project series exploring real-time biomechanical analysis without specialist hardware. Related to the Remaker body tracking work and the AR ghost trainer concept.
