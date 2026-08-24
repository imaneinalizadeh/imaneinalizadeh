"""
config.py

Central configuration for the Swift Bot Python client — camera index,
server connection details, and the emotion-to-movement-command mapping
table used by robot_client.py.
"""

# --- Camera / vision ---
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# --- Server connection (Java side) ---
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 5050
SOCKET_TIMEOUT_SECONDS = 5.0

# --- Emotion -> movement command mapping ---
# Only "happiness" and "anger" are mapped to movement commands per the
# original project spec — neutral/other emotions are treated as HOLD
# (no movement change) rather than guessing at an appropriate response.
EMOTION_COMMAND_MAP = {
    "happy": "ADVANCE",
    "angry": "RETREAT",
    "neutral": "HOLD",
    "sad": "HOLD",
    "surprise": "HOLD",
    "fear": "HOLD",
    "disgust": "HOLD",
}

DEFAULT_COMMAND = "HOLD"

# Minimum confidence (0-1) required before an emotion is acted on —
# below this, we send HOLD rather than react to a low-confidence guess.
EMOTION_CONFIDENCE_THRESHOLD = 0.4
