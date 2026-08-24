"""
ghost_mirror.py

Split-screen version of the ghost trainer: instead of a synthetic
wireframe skeleton, the right-hand pane plays back an actual recorded
"textbook" rep (extracted with extract_ghost_from_video.py), matched
frame-by-frame to the live feed's current phase percentage. This
reads as a real human demonstrating correct form next to you, rather
than an abstract skeleton overlay — the follow-up direction requested
after the wireframe version of ghost_trainer.py.

Requires: the reference video used to build the ghost table (so this
script can seek to the corresponding frame), plus the ghost table
itself (phase_pct -> knee_angle mapping) to know WHICH frame of the
reference video corresponds to the live feed's current phase.

Usage:
    python3 ghost_mirror.py --live path/to/my_attempt.mp4 \\
        --reference path/to/textbook_squat.mp4 \\
        --ghost-table data/ghost_reference/ghost_table_squat.json \\
        --headless --export results/mirror_session.json
"""

import argparse
import json

import cv2

from angle_calculator import calculate_angle
from ghost_reference import load_ghost_table
from pose_utils import PoseExtractor


def phase_to_reference_frame(phase_pct, reference_frame_count):
    """
    Maps a 0-100 phase percentage onto a frame index in the reference
    video, assuming the reference video's frames were captured evenly
    across its own single rep (true if the reference video was
    trimmed to exactly one rep before use).
    """
    idx = int(round(phase_pct / 100.0 * (reference_frame_count - 1)))
    return max(0, min(reference_frame_count - 1, idx))


def run(live_source, reference_path, ghost_table_path, headless, export_path,
        standing_angle=172.0, bottom_angle=80.0):
    ghost_table = load_ghost_table(ghost_table_path)

    live_cap = cv2.VideoCapture(live_source)
    ref_cap = cv2.VideoCapture(reference_path)
    if not live_cap.isOpened():
        raise RuntimeError(f"Could not open live source: {live_source}")
    if not ref_cap.isOpened():
        raise RuntimeError(f"Could not open reference video: {reference_path}")

    ref_frame_count = int(ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if ref_frame_count <= 0:
        raise RuntimeError("Reference video reports 0 frames — re-encode it or check the path.")

    extractor = PoseExtractor()
    frame_idx = 0
    log = []

    while True:
        ok, live_frame = live_cap.read()
        if not ok:
            break

        frame_rgb = cv2.cvtColor(live_frame, cv2.COLOR_BGR2RGB)
        landmarks = extractor.process(frame_rgb)

        ref_frame = None
        phase_pct = None

        if landmarks is not None:
            hip, knee, ankle = landmarks["hip"][:2], landmarks["knee"][:2], landmarks["ankle"][:2]
            live_knee_angle = calculate_angle(hip, knee, ankle)

            span = standing_angle - bottom_angle
            phase_pct = 0.0 if span == 0 else \
                100.0 * (standing_angle - live_knee_angle) / span
            phase_pct = max(0.0, min(100.0, phase_pct))

            ref_idx = phase_to_reference_frame(phase_pct, ref_frame_count)
            ref_cap.set(cv2.CAP_PROP_POS_FRAMES, ref_idx)
            ref_ok, ref_frame = ref_cap.read()
            if not ref_ok:
                ref_frame = None

            log.append({
                "frame": frame_idx,
                "phase_pct": round(phase_pct, 1),
                "live_knee_angle": round(live_knee_angle, 1),
                "reference_frame_index": ref_idx,
            })

        if not headless:
            h, w = live_frame.shape[:2]
            if ref_frame is not None:
                ref_resized = cv2.resize(ref_frame, (w, h))
            else:
                ref_resized = (live_frame * 0)  # blank pane if no detection yet
            combined = cv2.hconcat([live_frame, ref_resized])
            if phase_pct is not None:
                cv2.putText(combined, f"Phase: {phase_pct:.0f}%", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("Ghost Mirror — You | Reference", combined)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

    extractor.close()
    live_cap.release()
    ref_cap.release()
    if not headless:
        cv2.destroyAllWindows()

    if export_path:
        with open(export_path, "w") as f:
            json.dump(log, f, indent=2)
        print(f"Exported {len(log)} frames of phase-matched log to {export_path}")

    return log


def main():
    parser = argparse.ArgumentParser(description="Split-screen ghost mirror")
    parser.add_argument("--live", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--ghost-table", required=True)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--export", default=None)
    args = parser.parse_args()
    run(args.live, args.reference, args.ghost_table, args.headless, args.export)


if __name__ == "__main__":
    main()
