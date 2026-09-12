from __future__ import annotations

from pathlib import Path

import numpy as np
from django.test import SimpleTestCase

from .detection import Detection, HOGPersonDetector, build_detector
from .io import save_annotated_video, save_image, sample_frames
from .tracking import IOUTracker
from .visualize import render_heatmap, save_heatmap_png


def _solid_frame(width=100, height=100, color=(50, 50, 50)) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = color
    return frame


class DetectionTests(SimpleTestCase):
    def test_hog_detector_on_blank_frame_returns_no_detections(self):
        detector = HOGPersonDetector()
        detections = detector.detect(_solid_frame())
        self.assertEqual(detections, [])

    def test_hog_detector_short_circuits_below_window_size(self):
        detector = HOGPersonDetector()
        tiny = _solid_frame(width=10, height=10)
        self.assertEqual(detector.detect(tiny), [])

    def test_build_detector_hog(self):
        self.assertIsInstance(build_detector("hog"), HOGPersonDetector)

    def test_build_detector_unknown_backend_raises(self):
        with self.assertRaises(ValueError):
            build_detector("not-a-backend")

    def test_detection_geometry_properties(self):
        det = Detection(x1=10, y1=20, x2=30, y2=60, score=0.9)
        self.assertEqual(det.centroid, (20.0, 40.0))
        self.assertEqual(det.width, 20.0)
        self.assertEqual(det.height, 40.0)
        self.assertEqual(det.area, 800.0)


class TrackingTests(SimpleTestCase):
    def test_new_detection_spawns_a_track(self):
        tracker = IOUTracker()
        tracks = tracker.update([Detection(0, 0, 10, 10, 0.9)])
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].track_id, 1)

    def test_matching_detection_keeps_same_track_id(self):
        tracker = IOUTracker(iou_threshold=0.1)
        tracker.update([Detection(0, 0, 10, 10, 0.9)])
        tracks = tracker.update([Detection(1, 1, 11, 11, 0.9)])
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].track_id, 1)

    def test_track_survives_brief_occlusion_then_ages_out(self):
        tracker = IOUTracker(max_age=1)
        tracker.update([Detection(0, 0, 10, 10, 0.9)])
        tracker.update([])  # missed one frame -- track ages but survives
        self.assertIn(1, tracker.tracks)
        tracker.update([])  # missed a second frame -- exceeds max_age=1
        self.assertNotIn(1, tracker.tracks)

    def test_total_tracks_spawned_counts_dropped_tracks(self):
        tracker = IOUTracker(max_age=0)
        tracker.update([Detection(0, 0, 10, 10, 0.9)])
        tracker.update([Detection(500, 500, 510, 510, 0.9)])  # no overlap -> new track, old dropped
        self.assertEqual(tracker.total_tracks_spawned, 2)


class HeatmapTests(SimpleTestCase):
    def test_render_heatmap_bins_samples(self):
        grid = render_heatmap([(5, 5), (5, 5), (95, 95)], frame_width=100, frame_height=100, cell_size=20)
        self.assertEqual(grid.sample_count, 3)
        self.assertGreater(grid.counts.max(), 0)

    def test_render_heatmap_rejects_non_positive_frame_size(self):
        with self.assertRaises(ValueError):
            render_heatmap([], frame_width=0, frame_height=100)

    def test_save_heatmap_png_is_atomic_and_readable(self, tmp_path=None):
        import tempfile

        grid = render_heatmap([(5, 5)], frame_width=40, frame_height=40, cell_size=10)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "heatmap.png"
            save_heatmap_png(grid, out)
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 0)
            # no leftover temp file
            leftovers = [p for p in Path(tmp).iterdir() if p.name != "heatmap.png"]
            self.assertEqual(leftovers, [])


class PoseAvailabilityTests(SimpleTestCase):
    def test_mediapipe_without_solutions_attribute_raises_unavailable_error(self):
        """A successful `import mediapipe` doesn't guarantee the real Google
        package -- regression test for the case an unconstrained resolve on
        an unsupported Python version lands an unrelated PyPI package also
        named 'mediapipe' with no `.solutions`."""
        import sys
        import types
        import unittest.mock as mock

        from .pose import MediaPipePoseEstimator, MediaPipeUnavailableError

        fake_mediapipe = types.ModuleType("mediapipe")  # deliberately has no `.solutions`
        with mock.patch.dict(sys.modules, {"mediapipe": fake_mediapipe}):
            with self.assertRaises(MediaPipeUnavailableError):
                MediaPipePoseEstimator()


class IoTests(SimpleTestCase):
    def test_save_image_is_atomic_and_readable(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "frame.png"
            save_image(_solid_frame(), out)
            self.assertTrue(out.exists())
            leftovers = [p for p in Path(tmp).iterdir() if p.name != "frame.png"]
            self.assertEqual(leftovers, [])

    def test_save_annotated_video_writes_playable_file(self):
        import tempfile

        frames = [_solid_frame(color=(i, i, i)) for i in range(5)]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "clip.mp4"
            save_annotated_video(frames, out, fps=5.0)
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 0)

    def test_sample_frames_missing_file_raises(self):
        from .io import VideoReadError

        with self.assertRaises(VideoReadError):
            list(sample_frames("/no/such/video.mp4"))
