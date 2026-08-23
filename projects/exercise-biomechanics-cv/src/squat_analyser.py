"""
squat_analyser.py
Real-time squat form analysis using MediaPipe pose estimation.

Tracks knee angle and counts reps. Gives live pass/fail feedback
based on depth threshold (90 degrees).

Usage: python squat_analyser.py
Press 'q' to quit.
"""

import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def calculate_angle(a, b, c) -> float:
    """
    Calculate the angle at point b formed by points a-b-c.
    All points are (x, y) tuples in normalised or pixel coordinates.
    Returns angle in degrees.
    """
    a, b, c = np.array(a), np.array(b), np.array(c)
    v1 = a - b
    v2 = c - b
    cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def get_point(landmarks, idx, w, h):
    lm = landmarks[idx]
    return (int(lm.x * w), int(lm.y * h))


class SquatCounter:
    def __init__(self, depth_threshold: float = 90.0):
        self.depth_threshold = depth_threshold
        self.reps = 0
        self.stage = 'up'  # 'up' or 'down'

    def update(self, knee_angle: float):
        if knee_angle < self.depth_threshold:
            self.stage = 'down'
        if knee_angle > 160 and self.stage == 'down':
            self.stage = 'up'
            self.reps += 1


def main():
    cap = cv2.VideoCapture(0)
    counter = SquatCounter(depth_threshold=90.0)

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
                lm = results.pose_landmarks.landmark
                mp_drawing.draw_landmarks(frame, results.pose_landmarks,
                                          mp_pose.POSE_CONNECTIONS)

                # Left side landmarks
                hip   = get_point(lm, mp_pose.PoseLandmark.LEFT_HIP.value,   w, h)
                knee  = get_point(lm, mp_pose.PoseLandmark.LEFT_KNEE.value,  w, h)
                ankle = get_point(lm, mp_pose.PoseLandmark.LEFT_ANKLE.value, w, h)

                knee_angle = calculate_angle(hip, knee, ankle)
                counter.update(knee_angle)

                # Colour based on depth
                angle_col = (0, 255, 0) if knee_angle <= 90 else (0, 0, 255)

                # Draw angle at knee
                cv2.putText(frame, f'{knee_angle:.0f}°', knee,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, angle_col, 2)

                # Draw angle arc indicator
                cv2.circle(frame, knee, 8, angle_col, -1)

                # Overlay stats
                cv2.rectangle(frame, (0, 0), (250, 120), (0, 0, 0), -1)
                cv2.putText(frame, f'Reps: {counter.reps}', (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
                cv2.putText(frame, f'Stage: {counter.stage.upper()}', (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 255, 0) if counter.stage == 'down' else (200, 200, 200), 2)

                depth_txt = 'GOOD DEPTH' if knee_angle <= 90 else 'GO DEEPER'
                cv2.putText(frame, depth_txt, (10, 115),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, angle_col, 2)

            cv2.imshow('Squat Analyser', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
