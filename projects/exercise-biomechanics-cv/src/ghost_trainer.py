"""
ghost_trainer.py

Real-time AR overlay: draws a translucent "ghost" skeleton on top of
the live camera feed, showing where the lifter's joints SHOULD be at
their current point in the rep, so deviation from correct form is
visible as it happens rather than scored after the fact.

Follow-up project to squat_analyzer.py — see docs/architecture.md for
why this needed phase-based (not time-based) alignment between the
live skeleton and the ghost.

Usage:
    python3 ghost_trainer.py --video path/to.mp4 --headless --export results/ghost_session.json
"""

import argparse
import json

import cv2

from angle_calculator import calculate_angle
from ghost_reference import build_rule_based_ghost_table, lookup_ghost, load_ghost_table
from pose_utils import PoseExtractor

DEVIATION_GOOD = 8.0    # degrees — within this of the ghost = green
DEVIATION_WARN = 18.0   # degrees — within this = amber; beyond = red


def deviation_colour(delta_deg):
    d = abs(delta_deg)
    if d <= DEVIATION_GOOD:
        return (0, 200, 0)      # green (BGR)
    elif d <= DEVIATION_WARN:
        return (0, 200, 255)    # amber
    else:
        return (0, 0, 255)      # red


def run(video_source, headless, export_path, ghost_table_path,
        standing_knee_angle=172.0, bottom_knee_angle=80.0):
    if ghost_table_path:
        ghost_table = load_ghost_table(ghost_table_path)
    else:
        ghost_table = build_rule_based_ghost_table(
            standing_knee_angle=standing_knee_angle,
            bottom_knee_angle=bottom_knee_angle,
        )

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {video_source}")

    extractor = PoseExtractor()
    frame_idx = 0
    log = []

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

            live_knee_angle = calculate_angle(hip, knee, ankle)

            # Phase = how far through the rep the live angle implies,
            # mapped onto the same 0-100 scale the ghost table uses.
            span = standing_knee_angle - bottom_knee_angle
            phase_pct = 0.0 if span == 0 else \
                100.0 * (standing_knee_angle - live_knee_angle) / span
            phase_pct = max(0.0, min(100.0, phase_pct))

            ghost = lookup_ghost(ghost_table, phase_pct)
            deviation = live_knee_angle - ghost["knee_angle"]

            log.append({
                "frame": frame_idx,
                "phase_pct": round(phase_pct, 1),
                "live_knee_angle": round(live_knee_angle, 1),
                "ghost_knee_angle": round(ghost["knee_angle"], 1),
                "deviation_deg": round(deviation, 1),
            })

            if not headless:
                colour = deviation_colour(deviation)
                cv2.putText(frame_bgr,
                            f"You: {live_knee_angle:.0f} deg | Ghost: {ghost['knee_angle']:.0f} deg "
                            f"(dev {deviation:+.0f})",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2)
                cv2.putText(frame_bgr, f"Phase: {phase_pct:.0f}%",
                            (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.imshow("Ghost Trainer", frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        frame_idx += 1

    extractor.close()
    cap.release()
    if not headless:
        cv2.destroyAllWindows()

    if export_path:
        with open(export_path, "w") as f:
            json.dump(log, f, indent=2)
        print(f"Exported {len(log)} frames of ghost-comparison log to {export_path}")

    if log:
        mean_abs_dev = sum(abs(r["deviation_deg"]) for r in log) / len(log)
        print(f"Mean absolute deviation from ghost: {mean_abs_dev:.1f} deg over {len(log)} frames")
    return log


def main():
    parser = argparse.ArgumentParser(description="Real-time AR ghost trainer overlay")
    parser.add_argument("--video", default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--export", default=None)
    parser.add_argument("--ghost-table", default=None,
                         help="Path to a saved ghost table JSON (default: rule-based)")
    args = parser.parse_args()

    source = args.video if isinstance(args.video, str) else 0
    run(source, args.headless, args.export, args.ghost_table)


if __name__ == "__main__":
    main()
