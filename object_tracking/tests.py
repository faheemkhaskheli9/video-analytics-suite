from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from django.test import TestCase, override_settings

from .task import ObjectTrackingTask


def _write_test_clip(path: Path, num_frames: int = 10, size=(64, 64)) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, size)
    for i in range(num_frames):
        frame = np.full((size[1], size[0], 3), 200, dtype=np.uint8)
        cv2.rectangle(frame, (i % 20, 5), (i % 20 + 20, 60), (0, 0, 0), thickness=-1)
        writer.write(frame)
    writer.release()


class ObjectTrackingTaskTests(TestCase):
    @override_settings(VIDEO_FRAME_STRIDE=1, VIDEO_MAX_FRAMES=10, DETECTOR_BACKEND="hog")
    def test_run_produces_video_and_metrics_on_a_tiny_clip(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            clip_path = Path(tmp) / "clip.mp4"
            _write_test_clip(clip_path)
            output_dir = Path(tmp) / "output"
            output_dir.mkdir()

            task = ObjectTrackingTask()
            result = task.run(clip_path, output_dir, {})

            self.assertIn("frames_processed", result.metrics)
            self.assertEqual(result.metrics["frames_processed"], 10)
            self.assertEqual(result.visualization_paths, ["annotated.mp4"])
            self.assertTrue((output_dir / "annotated.mp4").exists())

    def test_task_is_registered(self):
        from video_core.registry import get_task, is_registered

        self.assertTrue(is_registered("object_tracking"))
        self.assertEqual(get_task("object_tracking").input_kind.value, "video")
