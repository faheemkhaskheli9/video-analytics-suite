"""A rule-based action classifier over MediaPipe pose keypoints.

Consolidated from the standalone ``human-action-recognition`` repo, which had
shipped pose extraction (Phase 1) but no classifier yet (Phase 2 was still
open). Rather than porting an unfinished/untrained classifier, this is an
explicit, documented heuristic over joint geometry and frame-to-frame hip
motion -- honestly a rule-based demo, not a trained action-recognition model
(see the task's summary and the README Limitations). It labels each sampled
frame as one of: walking, sitting, presenting, standing.
"""
from __future__ import annotations

from video_core.pose import PoseResult, joint_angle_degrees

LABELS = ("walking", "sitting", "presenting", "standing")

# A hip centroid moving more than this fraction of frame width between
# sampled frames counts as "walking".
_WALK_MOTION_THRESHOLD_FRAC = 0.02
# Hip-knee-ankle angle below this (degrees) means the knee is bent enough to
# call "sitting" (a standing leg is close to 180deg).
_SITTING_KNEE_ANGLE_MAX = 130.0


def classify_frame(
    pose: PoseResult,
    previous_hip_centroid: tuple[float, float] | None,
    width: int,
    height: int,
) -> str | None:
    """Returns one of ``LABELS``, or ``None`` if no person was detected."""
    if not pose.detected:
        return None

    left_wrist = pose.keypoint("left_wrist")
    right_wrist = pose.keypoint("right_wrist")
    left_shoulder = pose.keypoint("left_shoulder")
    right_shoulder = pose.keypoint("right_shoulder")
    if (left_wrist and left_shoulder and left_wrist.y < left_shoulder.y - 0.03) or (
        right_wrist and right_shoulder and right_wrist.y < right_shoulder.y - 0.03
    ):
        return "presenting"

    left_knee_angle = joint_angle_degrees(pose, "left_hip", "left_knee", "left_ankle", width, height)
    right_knee_angle = joint_angle_degrees(pose, "right_hip", "right_knee", "right_ankle", width, height)
    knee_angles = [a for a in (left_knee_angle, right_knee_angle) if a is not None]
    if knee_angles and min(knee_angles) < _SITTING_KNEE_ANGLE_MAX:
        return "sitting"

    left_hip = pose.keypoint("left_hip")
    right_hip = pose.keypoint("right_hip")
    if left_hip and right_hip and previous_hip_centroid is not None:
        hip_x = (left_hip.x + right_hip.x) / 2.0
        hip_y = (left_hip.y + right_hip.y) / 2.0
        dx = abs(hip_x - previous_hip_centroid[0])
        if dx > _WALK_MOTION_THRESHOLD_FRAC:
            return "walking"

    return "standing"


def hip_centroid(pose: PoseResult) -> tuple[float, float] | None:
    left_hip = pose.keypoint("left_hip")
    right_hip = pose.keypoint("right_hip")
    if left_hip is None or right_hip is None:
        return None
    return ((left_hip.x + right_hip.x) / 2.0, (left_hip.y + right_hip.y) / 2.0)
