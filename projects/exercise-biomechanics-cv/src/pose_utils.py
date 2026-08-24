"""
pose_utils.py

Thin wrapper around MediaPipe Pose for extracting the landmarks this
project actually needs (hip, knee, ankle, shoulder — left side by
default, since most webcam setups have the person roughly side-on).

NOTE ON MEDIAPIPE VERSION: pin to mediapipe==0.10.14. MediaPipe 1.0.0
dropped the legacy `mp.solutions` API entirely (`AttributeError: module
'mediapipe' has no attribute 'solutions'`), which every function below
depends on. See requirements.txt.
"""

import mediapipe as mp

mp_pose = mp.solutions.pose

# Landmark indices this project uses (MediaPipe's 33-point model)
LEFT_SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER.value
LEFT_HIP = mp_pose.PoseLandmark.LEFT_HIP.value
LEFT_KNEE = mp_pose.PoseLandmark.LEFT_KNEE.value
LEFT_ANKLE = mp_pose.PoseLandmark.LEFT_ANKLE.value


class PoseExtractor:
    """
    Wraps mp.solutions.pose.Pose so the rest of the codebase deals in
    plain (x, y, visibility) tuples rather than MediaPipe's protobuf
    landmark objects.
    """

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.pose = mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame_rgb):
        """
        frame_rgb: an RGB image (numpy array, H x W x 3).
        Returns a dict of {joint_name: (x, y, visibility)} in
        normalised [0, 1] image coordinates, or None if no person was
        detected in this frame.
        """
        result = self.pose.process(frame_rgb)
        if not result.pose_landmarks:
            return None

        lm = result.pose_landmarks.landmark
        return {
            "shoulder": (lm[LEFT_SHOULDER].x, lm[LEFT_SHOULDER].y, lm[LEFT_SHOULDER].visibility),
            "hip": (lm[LEFT_HIP].x, lm[LEFT_HIP].y, lm[LEFT_HIP].visibility),
            "knee": (lm[LEFT_KNEE].x, lm[LEFT_KNEE].y, lm[LEFT_KNEE].visibility),
            "ankle": (lm[LEFT_ANKLE].x, lm[LEFT_ANKLE].y, lm[LEFT_ANKLE].visibility),
        }

    def close(self):
        self.pose.close()
