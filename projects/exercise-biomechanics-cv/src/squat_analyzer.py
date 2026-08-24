"""
squat_analyzer.py

Real-time squat form analyser. Reads frames from a webcam (or a video
file, for reproducible testing without a camera), runs MediaPipe Pose,
computes knee/hip/torso angles, feeds the knee angle into RepCounter,
and overlays live feedback.

Usage:
    python3 squat_analyzer.py                      # webcam
    python3 squat_analyzer.py --video path/to.mp4  # video file
    python3 squat_analyzer.py --video path/to.mp4 --headless --export results/session.json

--headless skips the cv2.imshow() window (needed for CI / servers
without a display) and just processes the video, printing/exporting
the summary at the end.
"""

import argparse
import json
import time

import cv2

from angle_calculator import calculate_angle, torso_lean_angle
from pose_utils import PoseExtractor
from rep_counter import RepCounter

PARALLEL_KNEE_ANGLE = 95.0  # a knee angle at/below this = "parallel or below"


def run(video_source, headless, export_path, standing_angle, bottom_angle):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {video_source}")

    extractor = PoseExtractor()
    rep_counter = RepCounter(standing_angle=standing_angle, bottom_angle=bottom_angle)

    frame_idx = 0
    session_log = []
    t_start = time.time()

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        landmarks = extractor.process(frame_rgb)

        if landmarks is not None:
            hip = landmarks["hip"][:2]
            knee = landmarks["knee"][:2]
            ankle = landmarks["ankle"][:2]
            shoulder = landmarks["shoulder"][:2]

            knee_angle = calculate_angle(hip, knee, ankle)
            lean = torso_lean_angle(shoulder, hip)
            phase = rep_counter.update(knee_angle, frame_idx)

            session_log.append({
                "frame": frame_idx,
                "knee_angle_deg": round(knee_angle, 1),
                "torso_lean_deg": round(lean, 1),
                "phase": phase.value,
            })

            if not headless:
                depth_ok = knee_angle <= PARALLEL_KNEE_ANGLE
                colour = (0, 200, 0) if depth_ok else (0, 165, 255)
                cv2.putText(frame_bgr, f"Knee: {knee_angle:.0f} deg", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
                cv2.putText(frame_bgr, f"Reps: {rep_counter.rep_count}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame_bgr, f"Phase: {phase.value}", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.imshow("Squat Form Analyser", frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        frame_idx += 1

    elapsed = time.time() - t_start
    extractor.close()
    cap.release()
    if not headless:
        cv2.destroyAllWindows()

    summary = rep_counter.summary()
    summary["frames_processed"] = frame_idx
    summary["elapsed_seconds"] = round(elapsed, 2)

    print(json.dumps(summary, indent=2))

    if export_path:
        with open(export_path, "w") as f:
            json.dump({"summary": summary, "frame_log": session_log}, f, indent=2)
        print(f"Exported full session log to {export_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Real-time squat form analyser")
    parser.add_argument("--video", default=0, help="Path to a video file, or omit for webcam")
    parser.add_argument("--headless", action="store_true", help="Skip the display window")
    parser.add_argument("--export", default=None, help="Path to write full session JSON")
    parser.add_argument("--standing-angle", type=float, default=160.0)
    parser.add_argument("--bottom-angle", type=float, default=100.0)
    args = parser.parse_args()

    source = args.video if isinstance(args.video, str) else 0
    run(source, args.headless, args.export, args.standing_angle, args.bottom_angle)


if __name__ == "__main__":
    main()
