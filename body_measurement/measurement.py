"""Pure measurement-calculation logic, kept separate from I/O/detection so
it's unit-testable without a real detector or mediapipe model.

Consolidated from the standalone ``ai-body-measurement`` repo, which had
shipped the human-detection gate (Phase 1) but not yet the measurement math
(Phase 2). Scale calibration here uses an *assumed* average body height
(configurable), not a reference object in the photo, since this MVP has no
per-run calibration-object picker -- see the task's summary and README
Limitations.
"""
from __future__ import annotations

import math

from video_core.pose import PoseResult

ASSUMED_HEIGHT_CM = 170.0


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def pixels_per_cm(detection_box_height_px: float, assumed_height_cm: float = ASSUMED_HEIGHT_CM) -> float:
    if detection_box_height_px <= 0:
        raise ValueError("detection_box_height_px must be positive")
    return detection_box_height_px / assumed_height_cm


def compute_measurements(pose: PoseResult, width: int, height: int, px_per_cm: float) -> dict[str, float]:
    """Returns cm measurements for whichever of shoulder width / arm length /
    leg length have all the landmarks they need (missing landmarks are
    simply omitted from the result, never estimated from nothing)."""
    measurements: dict[str, float] = {}

    left_shoulder = pose.pixel_xy("left_shoulder", width, height)
    right_shoulder = pose.pixel_xy("right_shoulder", width, height)
    if left_shoulder and right_shoulder:
        measurements["shoulder_width_cm"] = round(_distance(left_shoulder, right_shoulder) / px_per_cm, 1)

    for side in ("left", "right"):
        shoulder = pose.pixel_xy(f"{side}_shoulder", width, height)
        elbow = pose.pixel_xy(f"{side}_elbow", width, height)
        wrist = pose.pixel_xy(f"{side}_wrist", width, height)
        if shoulder and elbow and wrist:
            arm_px = _distance(shoulder, elbow) + _distance(elbow, wrist)
            measurements[f"{side}_arm_length_cm"] = round(arm_px / px_per_cm, 1)

        hip = pose.pixel_xy(f"{side}_hip", width, height)
        knee = pose.pixel_xy(f"{side}_knee", width, height)
        ankle = pose.pixel_xy(f"{side}_ankle", width, height)
        if hip and knee and ankle:
            leg_px = _distance(hip, knee) + _distance(knee, ankle)
            measurements[f"{side}_leg_length_cm"] = round(leg_px / px_per_cm, 1)

    return measurements
