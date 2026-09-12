from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from django.test import TestCase, override_settings


def _write_test_clip(path: Path, size=(64, 64)) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, size)
    for _ in range(4):
        frame = np.full((size[1], size[0], 3), 220, dtype=np.uint8)
        writer.write(frame)
    writer.release()


class PoseEstimationTaskTests(TestCase):
    @override_settings(VIDEO_FRAME_STRIDE=1, VIDEO_MAX_FRAMES=4)
    def test_run_handles_a_clip_with_no_person(self):
        import tempfile

        from video_core.pose import MediaPipeUnavailableError

        from .task import PoseEstimationTask

        with tempfile.TemporaryDirectory() as tmp:
            clip_path = Path(tmp) / "clip.mp4"
            _write_test_clip(clip_path)
            output_dir = Path(tmp) / "output"
            output_dir.mkdir()
            try:
                result = PoseEstimationTask().run(clip_path, output_dir, {})
            except MediaPipeUnavailableError:
                self.skipTest("mediapipe not installed in this environment")
                return

            self.assertEqual(result.metrics["frames_with_person_detected"], 0)
            self.assertEqual(result.metrics["average_joint_angles_degrees"], {})
            self.assertTrue((output_dir / "annotated.mp4").exists())

    def test_task_is_registered(self):
        from video_core.registry import get_task, is_registered

        self.assertTrue(is_registered("pose_estimation"))
        self.assertEqual(get_task("pose_estimation").input_kind.value, "video")

    def test_missing_video_raises_video_task_error(self):
        from video_core.registry import VideoTaskError

        from .task import PoseEstimationTask
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(VideoTaskError):
                PoseEstimationTask().run(Path(tmp) / "missing.mp4", Path(tmp), {})
