"""
jump_analyser.py
Real-time jump height measurement using MediaPipe pose estimation.

Measures hang time from hip landmark trajectory and computes jump height
using the projectile motion formula: h = 0.5 * g * (t/2)^2

Usage: python jump_analyser.py
Press 'q' to quit.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import deque

G = 9.81  # gravitational acceleration m/s^2

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def calculate_jump_height(hang_time_s: float) -> float:
    """Calculate jump height in centimetres from hang time in seconds."""
    return 0.5 * G * (hang_time_s / 2) ** 2 * 100


def get_hip_y(landmarks, frame_h: int) -> float:
    """Return average y-coordinate of both hips in pixels."""
    left  = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y  * frame_h
    right = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * frame_h
    return (left + right) / 2


def get_ankle_y(landmarks, frame_h: int) -> float:
    """Return average y-coordinate of both ankles in pixels."""
    left  = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y  * frame_h
    right = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * frame_h
    return (left + right) / 2


class JumpAnalyser:
    def __init__(self, calibration_secs: int = 2):
        self.calibration_secs = calibration_secs
        self.standing_hip_y: float | None = None
        self.calibration_samples: list[float] = []
        self.calibration_start: float | None = None

        self.in_air = False
        self.takeoff_time: float | None = None
        self.jumps: deque = deque(maxlen=5)
        self.best_height_cm = 0.0

    def calibrate(self, hip_y: float, now: float) -> bool:
        """Collect standing samples. Returns True when calibration is done."""
        if self.calibration_start is None:
            self.calibration_start = now
        self.calibration_samples.append(hip_y)
        if now - self.calibration_start >= self.calibration_secs:
            self.standing_hip_y = float(np.mean(self.calibration_samples))
            return True
        return False

    def update(self, hip_y: float, ankle_y: float, now: float):
        """Detect jumps and record hang time."""
        if self.standing_hip_y is None:
            return

        # Threshold: hips rise more than 5% of standing position
        threshold = self.standing_hip_y * 0.95

        if not self.in_air and hip_y < threshold:
            self.in_air = True
            self.takeoff_time = now

        elif self.in_air and hip_y >= threshold:
            self.in_air = False
            if self.takeoff_time is not None:
                hang_time = now - self.takeoff_time
                if hang_time > 0.1:  # filter noise
                    height_cm = calculate_jump_height(hang_time)
                    self.jumps.append({
                        'hang_time': hang_time,
                        'height_cm': height_cm,
                    })
                    if height_cm > self.best_height_cm:
                        self.best_height_cm = height_cm


def draw_overlay(frame, analyser: JumpAnalyser, calibrated: bool, calibration_pct: float):
    h, w = frame.shape[:2]

    if not calibrated:
        pct = int(calibration_pct * 100)
        cv2.putText(frame, f'Calibrating... {pct}%', (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        cv2.rectangle(frame, (20, 70), (20 + int(400 * calibration_pct), 100),
                      (0, 255, 255), -1)
        return

    status_col = (0, 0, 255) if analyser.in_air else (0, 255, 0)
    status_txt = 'IN AIR' if analyser.in_air else 'GROUNDED'
    cv2.putText(frame, status_txt, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_col, 3)

    cv2.putText(frame, f'Jumps: {len(analyser.jumps)}', (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, f'Best: {analyser.best_height_cm:.1f} cm', (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 215, 255), 2)

    # Recent jumps list
    for idx, jump in enumerate(reversed(analyser.jumps)):
        y = 180 + idx * 35
        colour = (0, 255, 0) if jump['height_cm'] > 30 else (0, 165, 255)
        cv2.putText(frame, f'  Jump {len(analyser.jumps)-idx}: {jump["height_cm"]:.1f} cm '
                    f'({jump["hang_time"]:.3f}s)', (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2)

    # Bar chart
    if analyser.jumps:
        bar_x, bar_y = w - 220, 50
        max_h = max(j['height_cm'] for j in analyser.jumps) or 1
        for i, jump in enumerate(analyser.jumps):
            bar_height = int(jump['height_cm'] / max_h * 150)
            bx = bar_x + i * 40
            colour = (0, 255, 0) if jump['height_cm'] == analyser.best_height_cm else (0, 165, 255)
            cv2.rectangle(frame, (bx, bar_y + 150 - bar_height), (bx + 30, bar_y + 150),
                          colour, -1)
            cv2.putText(frame, f'{jump["height_cm"]:.0f}', (bx, bar_y + 165),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)


def main():
    cap = cv2.VideoCapture(0)
    analyser = JumpAnalyser(calibration_secs=2)
    calibrated = False

    with mp_pose.Pose(min_detection_confidence=0.7,
                      min_tracking_confidence=0.7) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks,
                                          mp_pose.POSE_CONNECTIONS)
                lm = results.pose_landmarks.landmark
                hip_y   = get_hip_y(lm, h)
                ankle_y = get_ankle_y(lm, h)
                now = time.time()

                if not calibrated:
                    elapsed = (now - analyser.calibration_start) if analyser.calibration_start else 0
                    calibrated = analyser.calibrate(hip_y, now)
                    pct = min(elapsed / analyser.calibration_secs, 1.0)
                    draw_overlay(frame, analyser, False, pct)
                else:
                    analyser.update(hip_y, ankle_y, now)
                    draw_overlay(frame, analyser, True, 1.0)
            else:
                cv2.putText(frame, 'No pose detected', (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow('Jump Analyser', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
