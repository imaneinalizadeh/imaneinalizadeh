"""
test_angle_calculator.py

Pure geometry tests — no MediaPipe or video needed, since
angle_calculator.py works on plain (x, y) tuples.
"""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from angle_calculator import calculate_angle, torso_lean_angle, depth_percentage


def test_right_angle():
    # a=(0,1), b=(0,0), c=(1,0) -> 90 degrees
    assert abs(calculate_angle((0, 1), (0, 0), (1, 0)) - 90.0) < 1e-6


def test_straight_line():
    # a and c on opposite sides of b -> 180 degrees
    assert abs(calculate_angle((0, 1), (0, 0), (0, -1)) - 180.0) < 1e-6


def test_zero_angle_when_points_coincide_direction():
    # a and c in the SAME direction from b -> 0 degrees
    assert abs(calculate_angle((0, 1), (0, 0), (0, 1)) - 0.0) < 1e-6


def test_degenerate_point_returns_zero_not_crash():
    # b coincides with a -> zero-length vector, must not raise
    assert calculate_angle((0, 0), (0, 0), (1, 0)) == 0.0


def test_known_squat_bottom_angle():
    # Roughly parallel squat: hip and ankle level, knee bent ~90 deg forward
    hip = (0.0, 0.5)
    knee = (0.15, 0.5)
    ankle = (0.0, 1.0)
    angle = calculate_angle(hip, knee, ankle)
    assert 60 < angle < 120  # sanity range, not an exact fixed value


def test_torso_lean_upright_is_near_zero():
    shoulder = (0.5, 0.0)
    hip = (0.5, 0.5)  # directly below shoulder -> upright
    assert torso_lean_angle(shoulder, hip) < 1.0


def test_torso_lean_increases_with_forward_lean():
    shoulder_upright = (0.5, 0.0)
    shoulder_leaning = (0.7, 0.0)  # shifted forward
    hip = (0.5, 0.5)
    upright = torso_lean_angle(shoulder_upright, hip)
    leaning = torso_lean_angle(shoulder_leaning, hip)
    assert leaning > upright


def test_depth_percentage_bounds():
    assert depth_percentage(hip_y=0.5, knee_y=0, standing_hip_y=0.5, bottom_hip_y=1.0) == 0.0
    assert depth_percentage(hip_y=1.0, knee_y=0, standing_hip_y=0.5, bottom_hip_y=1.0) == 100.0
    mid = depth_percentage(hip_y=0.75, knee_y=0, standing_hip_y=0.5, bottom_hip_y=1.0)
    assert abs(mid - 50.0) < 1e-6


def test_depth_percentage_clamped_outside_range():
    # values beyond the calibrated range should clamp, not extrapolate wildly
    assert depth_percentage(hip_y=2.0, knee_y=0, standing_hip_y=0.5, bottom_hip_y=1.0) == 100.0
    assert depth_percentage(hip_y=-1.0, knee_y=0, standing_hip_y=0.5, bottom_hip_y=1.0) == 0.0
