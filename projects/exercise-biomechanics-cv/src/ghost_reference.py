"""
ghost_reference.py

Builds and queries a "ghost" reference table: a phase-indexed sequence
of target joint angles representing correct squat form, used by both
ghost_trainer.py (synthetic rule-based ghost) and ghost_mirror.py
(real-video-derived ghost, built by extract_ghost_from_video.py).

A ghost table is a list of dicts, each shaped like:
    {"phase_pct": 0-100, "knee_angle": deg, "hip_angle": deg, "torso_lean": deg}

sorted by phase_pct. Looking a live frame up in this table means: find
the two entries bracketing the live rep's current phase_pct and
linearly interpolate, so the ghost animates smoothly regardless of how
coarsely the table was sampled.
"""

import json


def build_rule_based_ghost_table(n_points=21, bottom_knee_angle=80.0,
                                  standing_knee_angle=172.0,
                                  bottom_torso_lean=35.0):
    """
    Generates a synthetic "textbook form" ghost using a cosine
    ease-in/ease-out curve (natural human deceleration at the top and
    bottom of a rep, rather than a linear ramp which looks robotic)
    rather than a real recorded rep. Good enough as a v1 ghost /
    fallback when no recorded reference video is available.
    """
    import math
    table = []
    for i in range(n_points):
        phase_pct = 100.0 * i / (n_points - 1)
        # cosine interpolation 0->1 as phase goes 0->100
        t = (1 - math.cos(math.pi * phase_pct / 100.0)) / 2.0
        knee_angle = standing_knee_angle - t * (standing_knee_angle - bottom_knee_angle)
        torso_lean = t * bottom_torso_lean
        table.append({
            "phase_pct": round(phase_pct, 2),
            "knee_angle": round(knee_angle, 2),
            "torso_lean": round(torso_lean, 2),
        })
    return table


def lookup_ghost(table, phase_pct):
    """
    Linearly interpolate the ghost table at an arbitrary phase_pct
    (0-100). table must be sorted by phase_pct ascending.
    """
    phase_pct = max(0.0, min(100.0, phase_pct))

    if phase_pct <= table[0]["phase_pct"]:
        return dict(table[0])
    if phase_pct >= table[-1]["phase_pct"]:
        return dict(table[-1])

    for a, b in zip(table, table[1:]):
        if a["phase_pct"] <= phase_pct <= b["phase_pct"]:
            span = b["phase_pct"] - a["phase_pct"]
            frac = 0.0 if span == 0 else (phase_pct - a["phase_pct"]) / span
            out = {"phase_pct": phase_pct}
            for key in a:
                if key == "phase_pct":
                    continue
                out[key] = a[key] + frac * (b[key] - a[key])
            return out

    return dict(table[-1])  # fallback, shouldn't be reached


def save_ghost_table(table, path):
    with open(path, "w") as f:
        json.dump(table, f, indent=2)


def load_ghost_table(path):
    with open(path) as f:
        return json.load(f)
