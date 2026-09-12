"""Shared MediaPipe Pose wrapper, used by pose_estimation, action_recognition,
and body_measurement.

Ported/consolidated from ``human-pose-estimation``'s
``pose/mediapipe_estimator.py`` and ``human-action-recognition``'s
``pose/mediapipe_backend.py`` (both were near-identical MediaPipe wrappers
with a normalized ``Keypoint`` schema). ``mediapipe`` is imported lazily --
constructing this class is the first point at which it's required, so
importing ``video_core`` or the dashboard view never needs it installed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .registry import VideoTaskError

# MediaPipe Pose's 33 landmarks, in index order.
POSE_LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

# Bone pairs (by landmark name) used to draw the skeleton overlay.
POSE_SKELETON_EDGES = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("nose", "left_shoulder"), ("nose", "right_shoulder"),
]


class MediaPipeUnavailableError(VideoTaskError):
    """Raised when the mediapipe package isn't importable in this environment."""


@dataclass(frozen=True)
class Keypoint:
    name: str
    x: float  # normalized [0, 1], image width fraction
    y: float  # normalized [0, 1], image height fraction
    confidence: float


@dataclass(frozen=True)
class PoseResult:
    detected: bool
    keypoints: list[Keypoint] = field(default_factory=list)

    def keypoint(self, name: str) -> Keypoint | None:
        return next((kp for kp in self.keypoints if kp.name == name), None)

    def pixel_xy(self, name: str, width: int, height: int) -> tuple[float, float] | None:
        kp = self.keypoint(name)
        if kp is None:
            return None
        return (kp.x * width, kp.y * height)


class MediaPipePoseEstimator:
    def __init__(self, min_detection_confidence: float = 0.5, static_image_mode: bool = True):
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise MediaPipeUnavailableError(
                "mediapipe is not installed/importable in this environment. "
                "Install it (`pip install mediapipe`; already in "
                "requirements.txt) to use this feature."
            ) from exc
        if not hasattr(mp, "solutions"):
            # A successful `import mediapipe` does not guarantee the real
            # Google package: on an interpreter with no compatible mediapipe
            # wheel, an unconstrained resolve can land an unrelated PyPI
            # package that also happens to be named "mediapipe" instead of
            # failing loudly. Treat "imported but unusable" the same as "not
            # installed" rather than crashing deeper in with a confusing
            # AttributeError.
            raise MediaPipeUnavailableError(
                f"the installed 'mediapipe' package (version "
                f"{getattr(mp, '__version__', 'unknown')}) has no "
                "`mediapipe.solutions` -- this is not the real Google "
                "MediaPipe package, likely because this Python version has "
                "no compatible mediapipe wheel yet. Use Python < 3.13, or "
                "select a different task."
            )
        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            static_image_mode=static_image_mode,
            min_detection_confidence=min_detection_confidence,
        )

    def predict(self, image: np.ndarray) -> PoseResult:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self._pose.process(rgb)
        if not results.pose_landmarks:
            return PoseResult(detected=False, keypoints=[])
        keypoints = [
            Keypoint(name=name, x=lm.x, y=lm.y, confidence=lm.visibility)
            for name, lm in zip(POSE_LANDMARK_NAMES, results.pose_landmarks.landmark)
        ]
        return PoseResult(detected=True, keypoints=keypoints)

    def close(self) -> None:
        self._pose.close()

    def __enter__(self) -> "MediaPipePoseEstimator":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def joint_angle_degrees(
    pose: PoseResult, a: str, b: str, c: str, width: int, height: int
) -> float | None:
    """Angle at joint ``b`` formed by points ``a-b-c`` (e.g. elbow angle from
    shoulder-elbow-wrist), in degrees. ``None`` if any keypoint is missing."""
    pa, pb, pc = pose.pixel_xy(a, width, height), pose.pixel_xy(b, width, height), pose.pixel_xy(c, width, height)
    if pa is None or pb is None or pc is None:
        return None
    v1 = np.array(pa) - np.array(pb)
    v2 = np.array(pc) - np.array(pb)
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) or 1e-6
    cos_angle = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))
