"""Customer/traffic heatmap analytics, consolidated from the standalone
``video-traffic-heatmap`` repo: tracked-position aggregation into a
density heatmap, plus a whole-frame dwell-time estimate (seconds each track
was continuously present).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from django.conf import settings

from video_core.detection import build_detector
from video_core.io import sample_frames
from video_core.registry import BaseVideoTask, InputKind, TaskResult, VideoTaskError, register_task
from video_core.tracking import IOUTracker
from video_core.visualize import render_heatmap, save_heatmap_png


@register_task
class TrafficHeatmapTask(BaseVideoTask):
    key = "traffic_heatmap"
    label = "Traffic / Customer Heatmap"
    description = (
        "Tracks people across a clip and aggregates their positions into a "
        "density heatmap showing where time is spent in the frame, plus a "
        "per-track dwell-time summary."
    )
    input_kind = InputKind.VIDEO
    accepted_extensions = (".mp4", ".avi", ".mov", ".mkv")

    def run(self, input_path: Path, output_dir: Path, params: dict[str, Any]) -> TaskResult:
        detector = build_detector(settings.DETECTOR_BACKEND, device=settings.DETECTOR_DEVICE)
        tracker = IOUTracker()

        frames = list(sample_frames(input_path, frame_stride=settings.VIDEO_FRAME_STRIDE, max_frames=settings.VIDEO_MAX_FRAMES))
        if not frames:
            raise VideoTaskError("no frames could be read from the uploaded video")

        height, width = frames[0].image.shape[:2]
        samples: list[tuple[float, float]] = []
        frames_seen_per_track: dict[int, int] = {}
        for sample in frames:
            detections = detector.detect(sample.image)
            tracks = tracker.update(detections)
            for track in tracks:
                samples.append(track.centroid)
                frames_seen_per_track[track.track_id] = track.frames_seen

        grid = render_heatmap(samples, width, height)
        heatmap_path = save_heatmap_png(grid, output_dir / "heatmap.png")

        seconds_per_sampled_frame = settings.VIDEO_FRAME_STRIDE / 25.0  # demo-scale; see summary
        dwell_seconds = {
            f"track_{tid}": round(count * seconds_per_sampled_frame, 2)
            for tid, count in frames_seen_per_track.items()
        }
        top_dwell = dict(sorted(dwell_seconds.items(), key=lambda kv: kv[1], reverse=True)[:5])

        return TaskResult(
            metrics={
                "frames_processed": len(frames),
                "position_samples": grid.sample_count,
                "peak_cell_density": grid.to_json_dict()["peak_cell_count"],
                "unique_tracks": tracker.total_tracks_spawned,
                "top_5_dwell_seconds": top_dwell,
            },
            visualization_paths=[heatmap_path.name],
            summary=(
                f"{grid.sample_count} tracked-position samples binned into a "
                f"{grid.to_json_dict()['rows']}x{grid.to_json_dict()['cols']} "
                "density grid. Dwell seconds assume a 25fps source video "
                "(not measured from the file's real frame rate) -- a demo "
                "estimate, see Limitations. No historical-comparison store "
                "across runs in this MVP."
            ),
        )
