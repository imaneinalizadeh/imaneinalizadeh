# Architecture

```
Webcam frame
     |
     +--> FaceDetector (Haar cascade) --> face bounding box
     |
     +--> EmotionDetector (fer lib, or heuristic smile-cascade fallback)
     |         --> (label, confidence)
     |
     +--> GazeTracker (MediaPipe Face Mesh iris landmarks) --> left/centre/right
                |
                v
     emotion_to_command()  -- config.EMOTION_COMMAND_MAP, gated by
                               EMOTION_CONFIDENCE_THRESHOLD
                |
                v
     RobotClient -- TCP socket, de-duplicated command send
                |
        ======= network =======
                |
                v
     RobotServer (Java) -- accepts connection, reads lines
                |
                v
     CommandHandler -- parses "COMMAND|gaze", validates, de-dupes again
                |
                v
     MotorController -- the only class that would need to change to
                         drive a real robot instead of printing to stdout
```

## Why two emotion-detection backends

The original project used the `fer` library, which depends on
TensorFlow -- a large, slow-to-install dependency. `emotion_detector.py`
uses `fer` when available and transparently falls back to a Haar-cascade
smile detector otherwise (happy/neutral only -- no anger detection).
Both backends return the same `(label, confidence)` shape, so
`robot_client.py` doesn't need to know which is active.

## Why HOLD below a confidence threshold

A single flickery low-confidence frame (e.g. "angry" at 0.15
confidence -- essentially noise) could otherwise trigger a RETREAT.
`config.EMOTION_CONFIDENCE_THRESHOLD` requires a minimum confidence
before acting on any non-neutral emotion.

## Known limitations
- Single-face assumption: tracks whoever has the largest detected face.
- Gaze direction is computed but not currently acted on by the server
  -- logged for a future steering extension.
- The heuristic emotion fallback cannot detect anger at all.
