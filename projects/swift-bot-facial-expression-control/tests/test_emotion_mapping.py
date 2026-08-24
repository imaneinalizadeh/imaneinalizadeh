"""
test_emotion_mapping.py

Tests the emotion->command mapping logic (config.py + robot_client's
emotion_to_command) independently of any camera, MediaPipe model, or
socket connection.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python-client"))

import config
from robot_client import emotion_to_command


def test_happy_maps_to_advance():
    assert emotion_to_command("happy", 0.9) == "ADVANCE"


def test_angry_maps_to_retreat():
    assert emotion_to_command("angry", 0.9) == "RETREAT"


def test_neutral_maps_to_hold():
    assert emotion_to_command("neutral", 0.9) == "HOLD"


def test_unmapped_emotion_defaults_to_hold():
    assert emotion_to_command("surprise", 0.9) == "HOLD"


def test_low_confidence_forces_hold_even_for_happy():
    # Below EMOTION_CONFIDENCE_THRESHOLD, don't act on a shaky guess
    low_conf = config.EMOTION_CONFIDENCE_THRESHOLD - 0.1
    assert emotion_to_command("happy", low_conf) == "HOLD"


def test_confidence_at_threshold_boundary():
    # Exactly at threshold should NOT be treated as "below" it
    at_threshold = config.EMOTION_CONFIDENCE_THRESHOLD
    assert emotion_to_command("happy", at_threshold) == "ADVANCE"


def test_unknown_label_defaults_to_hold():
    assert emotion_to_command("some_future_label_not_in_map", 0.9) == "HOLD"
