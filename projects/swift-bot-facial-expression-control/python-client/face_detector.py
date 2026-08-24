"""
face_detector.py

Thin wrapper around OpenCV's bundled Haar Cascade classifier for face
detection. Kept separate from emotion_detector.py so the (cheap, fast)
face-presence check can gate the (more expensive) emotion model —
there's no point running emotion inference on a frame with no face.
"""

import cv2


class FaceDetector:
    def __init__(self, scale_factor=1.1, min_neighbors=5, min_size=(60, 60)):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError(f"Could not load Haar cascade from {cascade_path}")

        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    def detect(self, frame_bgr):
        """
        Returns a list of (x, y, w, h) bounding boxes, one per detected
        face, in pixel coordinates. Empty list if no face found.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
        )
        return [tuple(f) for f in faces]

    def largest_face(self, frame_bgr):
        """
        Convenience method: returns the single largest detected face
        (by bounding-box area), or None. Useful when exactly one
        person is expected to be interacting with the robot.
        """
        faces = self.detect(frame_bgr)
        if not faces:
            return None
        return max(faces, key=lambda f: f[2] * f[3])
