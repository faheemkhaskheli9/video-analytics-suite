from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from django.test import SimpleTestCase, TestCase, override_settings

from video_core.pose import Keypoint, PoseResult

from .classifier import classify_frame, hip_centroid


def _pose(**overrides: tuple[float, float, float]) -> PoseResult:
    """Build a standing pose (knees straight, wrists at hip height) with any
    named keypoint overridden by an (x, y, confidence) tuple."""
    defaults = {
        "left_shoulder": (0.4, 0.3, 1.0), "right_shoulder": (0.6, 0.3, 1.0),
        "left_hip": (0.42, 0.55, 1.0), "right_hip": (0.58, 0.55, 1.0),
        "left_knee": (0.42, 0.75, 1.0), "right_knee": (0.58, 0.75, 1.0),
        "left_ankle": (0.42, 0.95, 1.0), "right_ankle": (0.58, 0.95, 1.0),
        "left_wrist": (0.35, 0.5, 1.0), "right_wrist": (0.65, 0.5, 1.0),
    }
    defaults.update(overrides)
    keypoints = [Keypoint(name=name, x=x, y=y, confidence=c) for name, (x, y, c) in defaults.items()]
    return PoseResult(detected=True, keypoints=keypoints)


class ClassifierTests(SimpleTestCase):
    def test_no_person_returns_none(self):
        self.assertIsNone(classify_frame(PoseResult(detected=False), None, 100, 100))

    def test_bent_knee_classified_as_sitting(self):
        pose = _pose(left_knee=(0.42, 0.6, 1.0), left_ankle=(0.5, 0.6, 1.0))
        self.assertEqual(classify_frame(pose, None, 100, 100), "sitting")

    def test_raised_wrist_classified_as_presenting(self):
        pose = _pose(left_wrist=(0.4, 0.1, 1.0))
        self.assertEqual(classify_frame(pose, None, 100, 100), "presenting")

    def test_moving_hips_classified_as_walking(self):
        pose = _pose()
        previous_hip = (0.3, 0.55)  # far from this pose's ~0.5 hip x -> counts as motion
        self.assertEqual(classify_frame(pose, previous_hip, 100, 100), "walking")

    def test_stationary_standing_pose_classified_as_standing(self):
        pose = _pose()
        previous_hip = hip_centroid(pose)  # no motion
        self.assertEqual(classify_frame(pose, previous_hip, 100, 100), "standing")

    def test_hip_centroid_none_when_hip_missing(self):
        pose = PoseResult(detected=True, keypoints=[Keypoint(name="nose", x=0.5, y=0.1, confidence=1.0)])
        self.assertIsNone(hip_centroid(pose))


def _write_test_clip(path: Path, size=(64, 64)) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, size)
    for _ in range(4):
        frame = np.full((size[1], size[0], 3), 220, dtype=np.uint8)
        writer.write(frame)
    writer.release()


class ActionRecognitionTaskTests(TestCase):
    @override_settings(VIDEO_FRAME_STRIDE=1, VIDEO_MAX_FRAMES=4)
    def test_run_handles_a_clip_with_no_person(self):
        import tempfile

        from .task import ActionRecognitionTask
        from video_core.pose import MediaPipeUnavailableError

        with tempfile.TemporaryDirectory() as tmp:
            clip_path = Path(tmp) / "clip.mp4"
            _write_test_clip(clip_path)
            output_dir = Path(tmp) / "output"
            output_dir.mkdir()
            try:
                result = ActionRecognitionTask().run(clip_path, output_dir, {})
            except MediaPipeUnavailableError:
                self.skipTest("mediapipe not installed in this environment")
                return

            self.assertEqual(result.metrics["frames_with_person_detected"], 0)
            self.assertTrue((output_dir / "annotated.mp4").exists())

    def test_task_is_registered(self):
        from video_core.registry import get_task, is_registered

        self.assertTrue(is_registered("action_recognition"))
        self.assertEqual(get_task("action_recognition").input_kind.value, "video")
