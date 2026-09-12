"""Object line-crossing counter, consolidated from the standalone
``yolo-line-crossing-counter`` repo. The crossing line is fixed at the
frame's vertical midline (no per-run line editor in this MVP -- see
Limitations); a track counts as "in" when its centroid crosses left-to-right
and "out" for right-to-left.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from django.conf import settings

from video_core.detection import build_detector
from video_core.io import sample_frames, save_annotated_video
from video_core.registry import BaseVideoTask, InputKind, TaskResult, VideoTaskError, register_task
from video_core.tracking import IOUTracker
from video_core.visualize import draw_counts_banner, draw_line, draw_tracks


@register_task
class LineCrossingCounterTask(BaseVideoTask):
    key = "line_crossing_counter"
    label = "Line-Crossing Counter"
    description = (
        "Detection + tracking + a configurable counting line with "
        "direction awareness: in/out counts for objects crossing a fixed "
        "vertical line at the frame's midpoint."
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
        line_x = width // 2
        last_side: dict[int, bool] = {}  # track_id -> was left-of-line (True) last frame seen
        in_count = 0
        out_count = 0

        annotated_frames = []
        for sample in frames:
            detections = detector.detect(sample.image)
            tracks = tracker.update(detections)

            for track in tracks:
                cx, _cy = track.centroid
                is_left = cx < line_x
                previous = last_side.get(track.track_id)
                if previous is not None and previous != is_left:
                    if previous is True and is_left is False:
                        in_count += 1   # crossed left -> right
                    else:
                        out_count += 1  # crossed right -> left
                last_side[track.track_id] = is_left

            annotated = draw_tracks(sample.image, tracks)
            annotated = draw_line(annotated, (line_x, 0), (line_x, height))
            annotated = draw_counts_banner(annotated, f"in: {in_count}   out: {out_count}")
            annotated_frames.append(annotated)

        video_path = save_annotated_video(annotated_frames, output_dir / "annotated.mp4", fps=6.0)

        return TaskResult(
            metrics={
                "frames_processed": len(frames),
                "in_count": in_count,
                "out_count": out_count,
                "net_count": in_count - out_count,
                "line_position_x_px": line_x,
            },
            visualization_paths=[video_path.name],
            summary=(
                "The counting line is fixed at the frame's vertical midline "
                f"(x={line_x}px) -- there is no per-run line editor in this "
                "MVP, see Limitations. A dropped-and-regrown track can "
                "double-count a crossing under occlusion near the line."
            ),
        )
