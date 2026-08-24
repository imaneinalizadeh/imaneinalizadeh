# Architecture

## Pipeline overview

```
Video/webcam frame
      │
      ▼
PoseExtractor (MediaPipe Pose, pose_utils.py)
      │  33-landmark output → we use shoulder/hip/knee/ankle (left side)
      ▼
angle_calculator.py  →  knee angle, torso lean
      │
      ├──► rep_counter.py  →  phase (standing/descending/bottom/ascending),
      │                        rep count, per-rep depth
      │
      └──► ghost_reference.py  →  phase-indexed target angle
                 │
                 ▼
        deviation = live angle − ghost angle → colour-coded feedback
```

## Two generations of this project

1. **squat_analyzer.py** — the original: live angle + rep count +
   after-the-fact session summary. No reference to compare against
   in real time, just pass/fail per rep based on a fixed depth
   threshold.

2. **ghost_trainer.py / ghost_mirror.py / extract_ghost_from_video.py**
   — the follow-up: instead of scoring form only after a rep
   completes, overlay a live "ghost" showing what correct form looks
   like AT THE CURRENT INSTANT, so deviation is visible as it happens.

   The hard part of generation 2 wasn't the overlay rendering — it
   was **phase alignment**. Two people (or the same person on
   different reps) never move at exactly the same speed, so
   indexing the ghost by wall-clock time produces a ghost that's
   visibly ahead or behind the live feed within a second or two.
   The fix used throughout `ghost_reference.py`, `ghost_trainer.py`,
   and `ghost_mirror.py`: index the ghost by **phase percentage**
   (0% = standing, 100% = deepest point), derived from the live
   knee angle itself, not from time. This is why `lookup_ghost()`
   takes a `phase_pct` argument rather than a frame number.

## Two ways to build a ghost

- **Rule-based** (`ghost_reference.build_rule_based_ghost_table`):
  a cosine-interpolated curve between a standing angle and a target
  bottom angle. No recorded video needed — useful as a v1 / fallback,
  and for unit testing (deterministic, no MediaPipe dependency).
- **Data-driven** (`extract_ghost_from_video.py`): runs pose
  extraction on any video, uses the same `RepCounter` threshold-
  crossing logic the live analyser uses to find rep boundaries,
  picks whichever detected rep had the largest depth swing (deepest
  = cleanest reference), and resamples it onto the same 0-100% phase
  axis. This is what feeds `ghost_mirror.py`'s split-screen real-
  video ghost.

## Known limitations

- Single-side (left) landmarks only — a person turned side-on with
  their right side to the camera won't track correctly. A production
  version would check landmark visibility scores and pick whichever
  side is more visible.
- `ghost_mirror.py`'s frame-seeking (`cv2.CAP_PROP_POS_FRAMES`) is
  accurate on most codecs but can drift by a frame or two on some
  compressed formats with sparse keyframes — fine for a visual aid,
  not for frame-exact biomechanical measurement.
- No 3D — everything here is 2D image-plane angles. A camera angle
  that isn't roughly perpendicular to the plane of motion will distort
  the knee angle reading (foreshortening).
