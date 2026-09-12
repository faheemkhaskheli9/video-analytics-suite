"""Real-time object detection & tracking, consolidated from the standalone
``realtime-object-tracking`` repo's frame-source/detection scaffold: this
task adds the tracker and trajectory visualization that repo's README
described but hadn't reached yet (Phase 2/3), reusing ``video_core``'s
shared detector and ``IOUTracker`` instead of a second implementation.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from django.conf import settings

from video_core.detection import build_detector
from video_core.io import sample_frames, save_annotated_video
from video_core.registry import BaseVideoTask, InputKind, TaskResult, VideoTaskError, register_task
from video_core.tracking import IOUTracker
from video_core.visualize import draw_counts_banner, draw_tracks


@register_task
class ObjectTrackingTask(BaseVideoTask):
    key = "object_tracking"
    label = "Real-Time Object Tracking"
    description = (
        "YOLO/HOG detection + IoU tracking with persistent object IDs and "
        "trajectory overlays across frames, plus an FPS benchmark of the "
        "detect+track loop."
    )
    input_kind = InputKind.VIDEO
    accepted_extensions = (".mp4", ".avi", ".mov", ".mkv")

    def run(self, input_path: Path, output_dir: Path, params: dict[str, Any]) -> TaskResult:
        detector = build_detector(settings.DETECTOR_BACKEND, device=settings.DETECTOR_DEVICE)
        tracker = IOUTracker()

        frames = list(sample_frames(input_path, frame_stride=settings.VIDEO_FRAME_STRIDE, max_frames=settings.VIDEO_MAX_FRAMES))
        if not frames:
            raise VideoTaskError("no frames could be read from the uploaded video")

        annotated_frames = []
        start = time.perf_counter()
        for sample in frames:
            detections = detector.detect(sample.image)
            tracks = tracker.update(detections)
            annotated = draw_tracks(sample.image, tracks)
            banner = f"active tracks: {len(tracks)}  total spawned: {tracker.total_tracks_spawned}"
            annotated_frames.append(draw_counts_banner(annotated, banner))
        elapsed = max(1e-6, time.perf_counter() - start)
        processing_fps = round(len(frames) / elapsed, 2)

        video_path = save_annotated_video(annotated_frames, output_dir / "annotated.mp4", fps=6.0)

        track_lengths = [t.frames_seen for t in tracker.tracks.values()]
        avg_track_length = round(sum(track_lengths) / len(track_lengths), 2) if track_lengths else 0.0

        return TaskResult(
            metrics={
                "frames_processed": len(frames),
                "total_tracks_spawned": tracker.total_tracks_spawned,
                "still_active_at_end": len(tracker.active_tracks()),
                "average_track_length_frames": avg_track_length,
                "processing_fps": processing_fps,
            },
            visualization_paths=[video_path.name],
            summary=(
                f"Detection+tracking-only FPS (excludes video decode/encode): "
                f"{processing_fps} fps on CPU with the {settings.DETECTOR_BACKEND.upper()} "
                "backend, sampling every "
                f"{settings.VIDEO_FRAME_STRIDE} source frames. A dropped-and-"
                "regrown track gets a new ID (no re-identification), so "
                "'total tracks spawned' can overcount unique objects under "
                "occlusion -- see Limitations."
            ),
        )
