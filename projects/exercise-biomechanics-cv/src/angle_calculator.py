"""
angle_calculator.py

Pure-geometry joint angle math, deliberately kept independent of
MediaPipe so it can be unit-tested with plain (x, y) tuples instead of
needing a real pose estimation pipeline running.

All angles are returned in degrees, computed as the angle at vertex
`b` in the triangle a-b-c (i.e. the angle between vectors b->a and
b->c). For a knee angle, for example: a=hip, b=knee, c=ankle.
"""

import math


def calculate_angle(a, b, c):
    """
    Angle in degrees at point b, given three 2D points a, b, c.
    Each point is an (x, y) tuple (normalised or pixel coordinates —
    the angle is scale-invariant either way).
    """
    ax, ay = a
    bx, by = b
    cx, cy = c

    ba = (ax - bx, ay - by)
    bc = (cx - bx, cy - by)

    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.hypot(*ba)
    mag_bc = math.hypot(*bc)

    if mag_ba == 0 or mag_bc == 0:
        return 0.0

    cos_angle = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_angle))


def torso_lean_angle(shoulder, hip):
    """
    Angle of the torso from vertical, in degrees. 0 = perfectly
    upright. Useful as a proxy for excessive forward lean during a
    squat.
    """
    dx = shoulder[0] - hip[0]
    dy = shoulder[1] - hip[1]
    # angle from vertical axis
    return math.degrees(math.atan2(abs(dx), abs(dy)))


def depth_percentage(hip_y, knee_y, standing_hip_y, bottom_hip_y):
    """
    Maps the current hip height to a 0-100% depth scale, where 0% is
    standing and 100% is the deepest point recorded during
    calibration. Used to index into a phase-aligned "ghost" reference
    rather than relying on wall-clock time, since two people (or the
    same person on different reps) rarely move at the same speed.
    """
    total_range = bottom_hip_y - standing_hip_y
    if total_range == 0:
        return 0.0
    pct = (hip_y - standing_hip_y) / total_range
    return max(0.0, min(100.0, pct * 100.0))
