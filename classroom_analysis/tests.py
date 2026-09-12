from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from django.test import TestCase, override_settings

from video_core.detection import Detection

from .task import _most_central_detection


def _write_test_clip(path: Path, size=(64, 64)) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, size)
    for _ in range(4):
        frame = np.full((size[1], size[0], 3), 220, dtype=np.uint8)
        writer.write(frame)
    writer.release()


class MostCentralDetectionTests(TestCase):
    def test_returns_none_for_empty_list(self):
        self.assertIsNone(_most_central_detection([], width=100))

    def test_picks_detection_closest_to_center(self):
        near_edge = Detection(0, 0, 10, 10, 0.9)     # centroid x=5
        near_center = Detection(45, 0, 55, 10, 0.9)  # centroid x=50
        result = _most_central_detection([near_edge, near_center], width=100)
        self.assertIs(result, near_center)


class ClassroomAnalysisTaskTests(TestCase):
    @override_settings(VIDEO_FRAME_STRIDE=1, VIDEO_MAX_FRAMES=4, DETECTOR_BACKEND="hog")
    def test_run_handles_a_clip_with_no_person(self):
        import tempfile

        from video_core.pose import MediaPipeUnavailableError

        from .task import ClassroomAnalysisTask

        with tempfile.TemporaryDirectory() as tmp:
            clip_path = Path(tmp) / "clip.mp4"
            _write_test_clip(clip_path)
            output_dir = Path(tmp) / "output"
            output_dir.mkdir()
            try:
                result = ClassroomAnalysisTask().run(clip_path, output_dir, {})
            except MediaPipeUnavailableError:
                self.skipTest("mediapipe not installed in this environment")
                return

            self.assertEqual(result.metrics["frames_with_teacher_detected"], 0)
            self.assertTrue((output_dir / "annotated.mp4").exists())

    def test_task_is_registered(self):
        from video_core.registry import get_task, is_registered

        self.assertTrue(is_registered("classroom_analysis"))
        self.assertEqual(get_task("classroom_analysis").input_kind.value, "video")
