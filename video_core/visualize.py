"""Drawing/rendering helpers shared by every feature app.

Boxes, tracks, skeletons, and a traffic/dwell heatmap -- the heatmap
renderer is ported near-verbatim from ``retail-video-analytics``'s
``analytics/heatmap.py``. All file writes go through ``video_core.io``'s
atomic helpers.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .detection import Detection
from .io import save_image
from .pose import POSE_SKELETON_EDGES, PoseResult
from .tracking import Track

_BOX_COLOR = (60, 180, 75)      # BGR green
_LINE_COLOR = (0, 165, 255)     # BGR orange
_TRACK_PALETTE = [
    (66, 133, 244), (219, 68, 55), (244, 180, 0), (15, 157, 88),
    (171, 71, 188), (0, 172, 193), (255, 112, 67), (124, 179, 66),
]


def _track_color(track_id: int) -> tuple[int, int, int]:
    return _TRACK_PALETTE[track_id % len(_TRACK_PALETTE)]


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    annotated = frame.copy()
    for det in detections:
        p1, p2 = (int(det.x1), int(det.y1)), (int(det.x2), int(det.y2))
        cv2.rectangle(annotated, p1, p2, _BOX_COLOR, 2)
        cv2.putText(
            annotated, f"{det.label} {det.score:.2f}", (p1[0], max(0, p1[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, _BOX_COLOR, 1,
        )
    return annotated


def draw_tracks(frame: np.ndarray, tracks: list[Track], *, draw_trails: bool = True) -> np.ndarray:
    annotated = frame.copy()
    for track in tracks:
        color = _track_color(track.track_id)
        x1, y1, x2, y2 = (int(v) for v in track.bbox)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated, f"ID {track.track_id}", (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
        )
        if draw_trails and len(track.history) > 1:
            points = np.array(track.history, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(annotated, [points], isClosed=False, color=color, thickness=2)
    return annotated


def draw_line(frame: np.ndarray, p1: tuple[int, int], p2: tuple[int, int]) -> np.ndarray:
    annotated = frame.copy()
    cv2.line(annotated, p1, p2, _LINE_COLOR, 2)
    return annotated


def draw_counts_banner(frame: np.ndarray, text: str) -> np.ndarray:
    annotated = frame.copy()
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 34), (0, 0, 0), thickness=-1)
    cv2.putText(annotated, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return annotated


def draw_skeleton(frame: np.ndarray, pose: PoseResult, *, label: str = "") -> np.ndarray:
    annotated = frame.copy()
    if not pose.detected:
        cv2.putText(annotated, "no person detected", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return annotated
    height, width = annotated.shape[:2]
    for a, b in POSE_SKELETON_EDGES:
        pa, pb = pose.pixel_xy(a, width, height), pose.pixel_xy(b, width, height)
        if pa is None or pb is None:
            continue
        cv2.line(annotated, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), (0, 255, 255), 2)
    for kp in pose.keypoints:
        cv2.circle(annotated, (int(kp.x * width), int(kp.y * height)), 3, (0, 128, 255), -1)
    if label:
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 30), (0, 0, 0), thickness=-1)
        cv2.putText(annotated, label, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return annotated


# --- heatmap (ported from retail-video-analytics/analytics/heatmap.py) -----

Sample = tuple[float, float]


@dataclass(frozen=True)
class HeatmapGrid:
    counts: np.ndarray  # shape (rows, cols), float; raw sample counts per cell
    cell_size: int
    frame_width: int
    frame_height: int
    sample_count: int

    def to_json_dict(self) -> dict:
        return {
            "cell_size": self.cell_size,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "rows": int(self.counts.shape[0]),
            "cols": int(self.counts.shape[1]),
            "sample_count": self.sample_count,
            "peak_cell_count": float(self.counts.max()) if self.sample_count else 0.0,
        }


def render_heatmap(samples: list[Sample], frame_width: int, frame_height: int, *, cell_size: int = 20) -> HeatmapGrid:
    """Bin ``samples`` into a density grid. Samples outside the nominal frame
    are clamped to the edge rather than dropped -- a box can sit slightly
    outside the frame, and silently discarding it would understate an edge
    hotspot."""
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame_width and frame_height must be positive")
    cols = (frame_width + cell_size - 1) // cell_size
    rows = (frame_height + cell_size - 1) // cell_size
    counts = np.zeros((rows, cols), dtype=np.float64)
    for x, y in samples:
        cx = min(max(int(x // cell_size), 0), cols - 1)
        cy = min(max(int(y // cell_size), 0), rows - 1)
        counts[cy, cx] += 1.0
    return HeatmapGrid(counts=counts, cell_size=cell_size, frame_width=frame_width, frame_height=frame_height, sample_count=len(samples))


def heatmap_image(grid: HeatmapGrid, *, blur_sigma: float = 1.0) -> np.ndarray:
    counts = grid.counts
    if blur_sigma > 0:
        counts = cv2.GaussianBlur(counts, ksize=(0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
    peak = float(counts.max())
    normed = np.zeros_like(counts, dtype=np.uint8) if peak <= 0 else np.clip(counts / peak * 255.0, 0, 255).astype(np.uint8)
    upscaled = cv2.resize(
        normed, (grid.frame_width, grid.frame_height),
        interpolation=cv2.INTER_NEAREST if blur_sigma == 0 else cv2.INTER_LINEAR,
    )
    return cv2.applyColorMap(upscaled, cv2.COLORMAP_JET)


def save_heatmap_png(grid: HeatmapGrid, out_path: Path, *, blur_sigma: float = 1.0) -> Path:
    return save_image(heatmap_image(grid, blur_sigma=blur_sigma), out_path)
