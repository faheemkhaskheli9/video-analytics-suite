from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from django.conf import settings

from video_core.detection import build_detector
from video_core.io import sample_frames, save_annotated_video
from video_core.registry import BaseVideoTask, InputKind, TaskResult, VideoTaskError, register_task
from video_core.tracking import IOUTracker
from video_core.visualize import draw_counts_banner, draw_tracks, render_heatmap, save_heatmap_png

from .events import EventEngine
from .zones import ZoneManager, default_zones


@register_task
class RetailAnalyticsTask(BaseVideoTask):
    key = "retail_analytics"
    label = "Retail Video Analytics"
    description = (
        "Detection + tracking + zone/event logic on a store-camera clip: "
        "customer counting, entry/exit counts, dwell time, cashier-absence "
        "alerts, and a foot-traffic heatmap. Ported from the standalone "
        "retail-video-analytics repo's pipeline."
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
        zone_manager = ZoneManager(default_zones(width, height))
        seconds_per_frame = settings.VIDEO_FRAME_STRIDE / 25.0  # assume 25fps source; demo-scale, not measured
        events = EventEngine(zone_manager, seconds_per_frame=seconds_per_frame)

        annotated_frames = []
        for sample in frames:
            detections = detector.detect(sample.image)
            tracks = tracker.update(detections)
            events.process(sample.index, tracks)

            annotated = draw_tracks(sample.image, tracks)
            for zone in zone_manager.zones:
                pts = np.array(zone.polygon, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(annotated, [pts], isClosed=True, color=(255, 200, 0), thickness=2)
            banner = f"customers: {events.unique_customer_count()}  max-concurrent: {events.max_concurrent_customers}"
            annotated = draw_counts_banner(annotated, banner)
            annotated_frames.append(annotated)

        video_path = save_annotated_video(annotated_frames, output_dir / "annotated.mp4", fps=6.0)

        grid = render_heatmap(events.heatmap_samples, width, height)
        heatmap_path = save_heatmap_png(grid, output_dir / "heatmap.png")

        zone_counts = {zc.zone_name: {"entries": zc.entries, "exits": zc.exits} for zc in events.zone_counts()}
        avg_dwell = {
            name: round(
                sum(r.seconds for r in events.dwell_records() if r.zone_name == name)
                / max(1, len([r for r in events.dwell_records() if r.zone_name == name])),
                2,
            )
            for name in zone_counts
        }

        return TaskResult(
            metrics={
                "frames_processed": len(frames),
                "unique_customers": events.unique_customer_count(),
                "max_concurrent_customers": events.max_concurrent_customers,
                "zone_counts": zone_counts,
                "average_dwell_seconds": avg_dwell,
                "cashier_absence_alerts": len(events.cashier_alerts),
                "heatmap_samples": grid.sample_count,
            },
            visualization_paths=[video_path.name, heatmap_path.name],
            summary=(
                f"Processed {len(frames)} sampled frames (every "
                f"{settings.VIDEO_FRAME_STRIDE} source frames, offline "
                f"{settings.DETECTOR_BACKEND.upper()} detector). Zones are a "
                "fixed entrance/checkout layout scaled to this video's "
                "resolution, not manually drawn -- see Limitations."
            ),
        )
