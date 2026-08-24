"""
gaze_tracker.py

Estimates rough gaze direction (left / centre / right) from MediaPipe
Face Mesh iris landmarks, so the robot can distinguish "looking at the
robot" from "looking away" as a secondary signal alongside emotion.

Pinned to mediapipe==0.10.14 for the same reason as the biomechanics
project's pose_utils.py — later versions dropped the legacy
`mp.solutions` API this depends on.
"""

import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

# Iris landmark indices (requires refine_landmarks=True)
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE_CORNERS = (33, 133)   # (outer, inner)
RIGHT_EYE_CORNERS = (362, 263)


class GazeTracker:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def _iris_ratio(self, landmarks, iris_indices, corner_indices):
        outer_idx, inner_idx = corner_indices
        outer_x = landmarks[outer_idx].x
        inner_x = landmarks[inner_idx].x
        iris_x = sum(landmarks[i].x for i in iris_indices) / len(iris_indices)

        eye_width = inner_x - outer_x
        if eye_width == 0:
            return 0.5
        ratio = (iris_x - outer_x) / eye_width
        return max(0.0, min(1.0, ratio))

    def estimate_direction(self, frame_rgb, left_threshold=0.35, right_threshold=0.65):
        """
        Returns one of "left", "centre", "right", or None if no face
        was detected. Ratio is averaged across both eyes for
        robustness to single-eye tracking noise.
        """
        result = self.mesh.process(frame_rgb)
        if not result.multi_face_landmarks:
            return None

        landmarks = result.multi_face_landmarks[0].landmark
        left_ratio = self._iris_ratio(landmarks, LEFT_IRIS, LEFT_EYE_CORNERS)
        right_ratio = self._iris_ratio(landmarks, RIGHT_IRIS, RIGHT_EYE_CORNERS)
        avg_ratio = (left_ratio + right_ratio) / 2.0

        if avg_ratio < left_threshold:
            return "left"
        elif avg_ratio > right_threshold:
            return "right"
        return "centre"

    def close(self):
        self.mesh.close()
