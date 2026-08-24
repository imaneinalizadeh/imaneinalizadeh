"""
test_rep_counter.py

Tests the RepCounter state machine against synthetic knee-angle
signals — including the noisy-signal case that motivated switching
from local-minima detection to threshold-crossing logic (see the
module docstring in rep_counter.py).
"""

import math
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rep_counter import RepCounter, RepPhase


def _cosine_reps(n_reps, depth=80.0, standing=172.0, frames_per_rep=60):
    signal = []
    for _ in range(n_reps):
        for t in range(frames_per_rep):
            mid = (standing + depth) / 2
            amp = (standing - depth) / 2
            signal.append(mid + amp * math.cos(2 * math.pi * t / frames_per_rep))
    return signal


def test_counts_clean_reps_correctly():
    rc = RepCounter(standing_angle=160, bottom_angle=100)
    signal = _cosine_reps(n_reps=5)
    for i, a in enumerate(signal):
        rc.update(a, i)
    assert rc.rep_count == 5


def test_counts_correctly_with_realistic_noise():
    """
    The regression test for the bug this module's docstring describes:
    local-minima detection over-counted on noisy signals. Threshold
    crossing should be immune to jitter this size.
    """
    random.seed(7)
    rc = RepCounter(standing_angle=160, bottom_angle=100)
    signal = _cosine_reps(n_reps=3)
    noisy = [a + random.uniform(-3, 3) for a in signal]
    for i, a in enumerate(noisy):
        rc.update(a, i)
    assert rc.rep_count == 3


def test_partial_rep_not_counted():
    """A descent that goes back up WITHOUT reaching bottom_angle must not count."""
    rc = RepCounter(standing_angle=160, bottom_angle=100)
    # Dips to 130 (below standing, above bottom) then back to 170 — not a full rep
    signal = [172, 160, 145, 130, 145, 160, 172]
    for i, a in enumerate(signal):
        rc.update(a, i)
    assert rc.rep_count == 0


def test_depth_tracking_records_minimum_angle():
    rc = RepCounter(standing_angle=160, bottom_angle=100)
    signal = _cosine_reps(n_reps=1, depth=75.0)
    for i, a in enumerate(signal):
        rc.update(a, i)
    assert rc.rep_count == 1
    assert abs(rc.rep_depths[0] - 75.0) < 1.0


def test_summary_success_rate():
    # bottom_angle=140 is loose enough that BOTH reps below count as
    # completed reps, but only the depth=80 one is "parallel or below"
    # (<=95 deg) — this isolates depth-quality tracking from the
    # separate concept of whether a rep was completed at all.
    rc = RepCounter(standing_angle=160, bottom_angle=140)
    deep = _cosine_reps(n_reps=1, depth=80.0)
    shallow = _cosine_reps(n_reps=1, depth=130.0)
    frame = 0
    for a in deep + shallow:
        rc.update(a, frame)
        frame += 1
    summary = rc.summary()
    assert summary["total_reps"] == 2
    assert summary["reps_at_parallel_or_below"] == 1
    assert summary["success_rate_pct"] == 50.0


def test_phase_starts_standing():
    rc = RepCounter()
    assert rc.phase == RepPhase.STANDING
