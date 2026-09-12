from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from django.test import TestCase, override_settings

from .task import LineCrossingCounterTask


def _write_moving_box_clip(path: Path, size=(64, 64)) -> None:
    """A box that starts left of center and ends right of center -- exactly
    one left-to-right crossing."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, size)
    for i in range(10):
        frame = np.full((size[1], size[0], 3), 220, dtype=np.uint8)
        x = 2 + i * 6  # sweeps from x=2 to x=56 across a 64px-wide frame
        cv2.rectangle(frame, (x, 5), (x + 20, 60), (0, 0, 0), thickness=-1)
        writer.write(frame)
    writer.release()


class LineCrossingCounterTaskTests(TestCase):
    @override_settings(VIDEO_FRAME_STRIDE=1, VIDEO_MAX_FRAMES=10, DETECTOR_BACKEND="hog")
    def test_run_produces_counts_and_video(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            clip_path = Path(tmp) / "clip.mp4"
            _write_moving_box_clip(clip_path)
            output_dir = Path(tmp) / "output"
            output_dir.mkdir()

            task = LineCrossingCounterTask()
            result = task.run(clip_path, output_dir, {})

            self.assertIn("in_count", result.metrics)
            self.assertIn("out_count", result.metrics)
            self.assertTrue((output_dir / "annotated.mp4").exists())

    def test_task_is_registered(self):
        from video_core.registry import get_task, is_registered

        self.assertTrue(is_registered("line_crossing_counter"))
        self.assertEqual(get_task("line_crossing_counter").accepted_extensions, (".mp4", ".avi", ".mov", ".mkv"))
