from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from django.test import SimpleTestCase, TestCase

from video_core.pose import Keypoint, PoseResult

from .measurement import compute_measurements, pixels_per_cm


class MeasurementMathTests(SimpleTestCase):
    def test_pixels_per_cm_rejects_non_positive_height(self):
        with self.assertRaises(ValueError):
            pixels_per_cm(0)

    def test_pixels_per_cm_scales_linearly(self):
        self.assertAlmostEqual(pixels_per_cm(170.0, assumed_height_cm=170.0), 1.0)
        self.assertAlmostEqual(pixels_per_cm(340.0, assumed_height_cm=170.0), 2.0)

    def test_compute_measurements_omits_missing_landmarks(self):
        pose = PoseResult(detected=True, keypoints=[Keypoint(name="nose", x=0.5, y=0.1, confidence=1.0)])
        measurements = compute_measurements(pose, 100, 100, px_per_cm=1.0)
        self.assertEqual(measurements, {})

    def test_compute_measurements_shoulder_width(self):
        keypoints = [
            Keypoint(name="left_shoulder", x=0.3, y=0.3, confidence=1.0),
            Keypoint(name="right_shoulder", x=0.7, y=0.3, confidence=1.0),
        ]
        pose = PoseResult(detected=True, keypoints=keypoints)
        # 100px wide frame: shoulders are 40px apart; px_per_cm=2 -> 20cm
        measurements = compute_measurements(pose, 100, 100, px_per_cm=2.0)
        self.assertEqual(measurements["shoulder_width_cm"], 20.0)


def _write_blank_image(path: Path) -> None:
    cv2.imwrite(str(path), np.full((100, 100, 3), 220, dtype=np.uint8))


class BodyMeasurementTaskTests(TestCase):
    def test_no_person_in_blank_image_raises(self):
        import tempfile

        from .task import BodyMeasurementTask, NoPersonDetectedError

        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "blank.png"
            _write_blank_image(img_path)
            output_dir = Path(tmp) / "output"
            output_dir.mkdir()
            with self.assertRaises(NoPersonDetectedError):
                BodyMeasurementTask().run(img_path, output_dir, {})

    def test_task_is_registered(self):
        from video_core.registry import get_task, is_registered

        self.assertTrue(is_registered("body_measurement"))
        self.assertEqual(get_task("body_measurement").input_kind.value, "image")
