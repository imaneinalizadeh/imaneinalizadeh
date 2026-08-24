"""
extract_ghost_from_video.py

Given ANY video of someone squatting (a downloaded tutorial clip, a
personal "textbook" rep, etc.), automatically finds the best complete
standing -> bottom -> standing cycle, normalises it to a phase-indexed
ghost table (0-100%), and saves it in the same format ghost_trainer.py
and ghost_mirror.py consume.

Pipeline:
1. Run MediaPipe pose on every frame, extract knee angle per frame.
2. Use RepCounter's threshold-crossing logic to find rep boundaries
   (reusing the exact same logic as live analysis, for consistency).
3. Pick the rep with the largest depth swing (standing_angle - min
   angle) as the "cleanest" rep — a shallow/lazy rep in the source
   video would make a poor reference.
4. Resample that rep's frames onto a uniform 0-100% phase axis (since
   the source video's frame rate and rep duration are unlikely to
   match any future comparison video).

Usage:
    python3 extract_ghost_from_video.py --video reference_squat.mp4 \\
        --out data/ghost_reference/ghost_table_squat.json
"""

import argparse
import json

import cv2

from angle_calculator import calculate_angle
from pose_utils import PoseExtractor
from rep_counter import RepCounter


def extract_knee_angle_sequence(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    extractor = PoseExtractor()
    angles = []  # None where no pose detected, to keep frame indices aligned
    frame_idx = 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        landmarks = extractor.process(frame_rgb)
        if landmarks is not None:
            hip, knee, ankle = landmarks["hip"][:2], landmarks["knee"][:2], landmarks["ankle"][:2]
            angles.append(calculate_angle(hip, knee, ankle))
        else:
            angles.append(None)
        frame_idx += 1

    extractor.close()
    cap.release()
    return angles


def find_best_rep(angles, standing_angle=160.0, bottom_angle=100.0):
    """
    Runs the frames through RepCounter and returns the (start, end)
    frame range of whichever detected rep had the largest depth swing.
    Frames with no detection (None) are treated as a hold at the last
    known angle, so brief tracking dropouts don't fragment a rep.
    """
    rc = RepCounter(standing_angle=standing_angle, bottom_angle=bottom_angle)
    last_known = standing_angle
    for i, a in enumerate(angles):
        angle = a if a is not None else last_known
        if a is not None:
            last_known = a
        rc.update(angle, i)

    if not rc.rep_frame_ranges:
        return None

    swings = [standing_angle - depth for depth in rc.rep_depths]
    best_idx = max(range(len(swings)), key=lambda i: swings[i])
    return rc.rep_frame_ranges[best_idx], rc.rep_depths[best_idx]


def normalise_to_ghost_table(angles, frame_range, standing_angle, n_points=21):
    start, end = frame_range
    segment = angles[start:end + 1]
    # fill any None gaps by holding the previous value
    filled = []
    last = standing_angle
    for a in segment:
        if a is not None:
            last = a
        filled.append(last)

    n = len(filled)
    table = []
    for i in range(n_points):
        phase_pct = 100.0 * i / (n_points - 1)
        # map phase back to a source-frame index (nearest-neighbour resample)
        src_idx = min(n - 1, int(round(phase_pct / 100.0 * (n - 1))))
        table.append({
            "phase_pct": round(phase_pct, 2),
            "knee_angle": round(filled[src_idx], 2),
        })
    return table


def main():
    parser = argparse.ArgumentParser(description="Extract a ghost reference rep from a video")
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--standing-angle", type=float, default=160.0)
    parser.add_argument("--bottom-angle", type=float, default=100.0)
    parser.add_argument("--points", type=int, default=21)
    args = parser.parse_args()

    print(f"Extracting knee-angle sequence from {args.video} ...")
    angles = extract_knee_angle_sequence(args.video)
    detected = sum(1 for a in angles if a is not None)
    print(f"  {len(angles)} frames, pose detected in {detected} of them")

    result = find_best_rep(angles, args.standing_angle, args.bottom_angle)
    if result is None:
        raise SystemExit("No complete rep detected in this video — try a clearer clip "
                          "or loosen --standing-angle / --bottom-angle.")

    (start, end), depth = result
    swing = args.standing_angle - depth
    print(f"  Best rep: frames {start}-{end} ({end - start + 1} frames), "
          f"depth {depth:.1f} deg, swing {swing:.1f} deg")

    table = normalise_to_ghost_table(angles, (start, end), args.standing_angle, args.points)
    with open(args.out, "w") as f:
        json.dump(table, f, indent=2)
    print(f"Saved {len(table)}-point ghost table to {args.out}")


if __name__ == "__main__":
    main()
