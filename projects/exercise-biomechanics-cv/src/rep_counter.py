"""
rep_counter.py

State machine for counting squat reps from a stream of knee angles.
Deliberately uses threshold-crossing logic rather than naive local
minima detection.

WHY THRESHOLD-CROSSING, NOT LOCAL MINIMA: an earlier version walked
the knee-angle signal looking for local minima (a frame where the
angle is lower than both neighbours) to mark "bottom of squat". On
real webcam data this is noisy — MediaPipe's per-frame jitter creates
dozens of spurious tiny local minima even when someone is holding
still at the bottom of a rep, causing wildly over-counted reps. The
fix: define a STANDING threshold and a BOTTOM threshold; a rep is
only counted after the signal crosses from above-standing to
below-bottom and back to above-standing again. This makes single-
frame noise irrelevant unless it's big enough to actually cross a
threshold, which real jitter isn't.
"""

from enum import Enum


class RepPhase(Enum):
    STANDING = "standing"
    DESCENDING = "descending"
    BOTTOM = "bottom"
    ASCENDING = "ascending"


class RepCounter:
    def __init__(self, standing_angle=160.0, bottom_angle=100.0):
        """
        standing_angle: knee angle (degrees) above which the person is
            considered fully standing.
        bottom_angle: knee angle below which the person is considered
            to have reached squat depth.
        """
        self.standing_angle = standing_angle
        self.bottom_angle = bottom_angle
        self.phase = RepPhase.STANDING
        self.rep_count = 0
        self.rep_depths = []          # min knee angle reached, per rep
        self.rep_frame_ranges = []    # (start_frame, end_frame) per rep
        self._current_min_angle = None
        self._rep_start_frame = None

    def update(self, knee_angle, frame_idx):
        """
        Feed one frame's knee angle in. Returns the current RepPhase.
        A rep is finalised (rep_count incremented) the moment the
        signal returns to STANDING after having touched BOTTOM.
        """
        if self.phase == RepPhase.STANDING:
            if knee_angle < self.standing_angle:
                self.phase = RepPhase.DESCENDING
                self._rep_start_frame = frame_idx
                self._current_min_angle = knee_angle

        elif self.phase == RepPhase.DESCENDING:
            self._current_min_angle = min(self._current_min_angle, knee_angle)
            if knee_angle < self.bottom_angle:
                self.phase = RepPhase.BOTTOM
            elif knee_angle > self.standing_angle:
                # false start — went back up without reaching depth
                self.phase = RepPhase.STANDING
                self._rep_start_frame = None

        elif self.phase == RepPhase.BOTTOM:
            self._current_min_angle = min(self._current_min_angle, knee_angle)
            if knee_angle > self.bottom_angle:
                self.phase = RepPhase.ASCENDING

        elif self.phase == RepPhase.ASCENDING:
            if knee_angle > self.standing_angle:
                # Rep complete
                self.rep_count += 1
                self.rep_depths.append(self._current_min_angle)
                self.rep_frame_ranges.append((self._rep_start_frame, frame_idx))
                self.phase = RepPhase.STANDING
                self._current_min_angle = None
                self._rep_start_frame = None
            elif knee_angle < self.bottom_angle:
                # dipped back down — still in the same rep
                self.phase = RepPhase.BOTTOM
                self._current_min_angle = min(self._current_min_angle, knee_angle)

        return self.phase

    def summary(self):
        good_depth_count = sum(1 for d in self.rep_depths if d <= 95.0)
        return {
            "total_reps": self.rep_count,
            "reps_at_parallel_or_below": good_depth_count,
            "success_rate_pct": (
                round(100.0 * good_depth_count / self.rep_count, 1)
                if self.rep_count else 0.0
            ),
            "best_depth_deg": round(min(self.rep_depths), 1) if self.rep_depths else None,
            "average_depth_deg": (
                round(sum(self.rep_depths) / len(self.rep_depths), 1)
                if self.rep_depths else None
            ),
            "rep_depths_deg": [round(d, 1) for d in self.rep_depths],
        }
