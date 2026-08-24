"""
emotion_detector.py

Facial emotion classification. Tries to use the `fer` library (FER-2013
CNN weights) if it's installed; falls back to a lightweight geometric
heuristic (mouth aspect ratio via a Haar smile cascade, as a crude
happy/not-happy signal) if `fer`/its TensorFlow dependency isn't
available. The fallback exists so this module — and everything built
on top of it (robot_client.py, the emotion->command mapping) — stays
testable and demoable without requiring a multi-hundred-MB TensorFlow
install.

Either backend returns the same shape: (label: str, confidence: float).
"""

import cv2

try:
    from fer import FER as _FERModel
    _HAS_FER = True
except ImportError:
    _HAS_FER = False


class EmotionDetector:
    def __init__(self, prefer_fer=True):
        self.backend = "fer" if (prefer_fer and _HAS_FER) else "heuristic"

        if self.backend == "fer":
            self._fer = _FERModel(mtcnn=False)
        else:
            smile_path = cv2.data.haarcascades + "haarcascade_smile.xml"
            self._smile_cascade = cv2.CascadeClassifier(smile_path)

    def detect(self, frame_bgr, face_box=None):
        """
        face_box: optional (x, y, w, h) from FaceDetector, to crop
        before running the heuristic backend (the fer library does its
        own face detection internally if face_box is None).

        Returns (label, confidence) where label is one of:
        "happy", "angry", "neutral", "sad", "surprise", "fear", "disgust"
        """
        if self.backend == "fer":
            return self._detect_fer(frame_bgr)
        return self._detect_heuristic(frame_bgr, face_box)

    def _detect_fer(self, frame_bgr):
        result = self._fer.top_emotion(frame_bgr)
        if result is None or result[0] is None:
            return "neutral", 0.0
        label, confidence = result
        return label, float(confidence)

    def _detect_heuristic(self, frame_bgr, face_box):
        """
        Crude fallback: detect a smile within the face region using
        OpenCV's Haar smile cascade. Presence of a smile -> "happy" at
        a fixed moderate confidence; absence -> "neutral". This cannot
        distinguish anger/sadness/etc. at all — it exists purely so
        the rest of the pipeline (movement mapping, socket protocol)
        can be built and tested without the `fer` + TensorFlow
        dependency installed.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if face_box is not None:
            x, y, w, h = face_box
            roi = gray[y:y + h, x:x + w]
        else:
            roi = gray

        if roi.size == 0:
            return "neutral", 0.0

        smiles = self._smile_cascade.detectMultiScale(
            roi, scaleFactor=1.7, minNeighbors=20, minSize=(25, 25)
        )

        if len(smiles) > 0:
            return "happy", 0.6
        return "neutral", 0.5
