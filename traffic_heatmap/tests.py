from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from django.test import TestCase, override_settings

from .task import TrafficHeatmapTask


def _write_test_clip(path: Path, size=(64, 64)) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, size)
    for i in range(8):
        frame = np.full((size[1], size[0], 3), 210, dtype=np.uint8)
        cv2.rectangle(frame, (10, 5), (30, 60), (0, 0, 0), thickness=-1)
        writer.write(frame)
    writer.release()


class TrafficHeatmapTaskTests(TestCase):
    @override_settings(VIDEO_FRAME_STRIDE=1, VIDEO_MAX_FRAMES=8, DETECTOR_BACKEND="hog")
    def test_run_produces_heatmap_and_metrics(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            clip_path = Path(tmp) / "clip.mp4"
            _write_test_clip(clip_path)
            output_dir = Path(tmp) / "output"
            output_dir.mkdir()

            result = TrafficHeatmapTask().run(clip_path, output_dir, {})

            self.assertEqual(result.visualization_paths, ["heatmap.png"])
            self.assertTrue((output_dir / "heatmap.png").exists())
            self.assertIn("position_samples", result.metrics)

    def test_task_is_registered(self):
        from video_core.registry import get_task, is_registered

        self.assertTrue(is_registered("traffic_heatmap"))
        self.assertEqual(get_task("traffic_heatmap").input_kind.value, "video")
