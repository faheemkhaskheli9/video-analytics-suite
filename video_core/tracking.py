"""A minimal IoU-based multi-object tracker, shared by every feature app.

Ported near-verbatim from ``retail-video-analytics``'s
``tracking/tracker.py``. This is a simplified SORT-style tracker: no Kalman
filter, just greedy IoU matching between the previous frame's tracks and the
current frame's detections, with track aging so brief missed detections do
not immediately kill a track. It is intentionally dependency-light (numpy
only) so the MVP runs anywhere the detector backend runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .detection import Detection


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Intersection-over-union of two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    track_id: int
    detection: Detection
    hits: int = 1
    age: int = 0            # frames since last matched detection
    frames_seen: int = 1
    history: list[tuple[float, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.history.append(self.detection.centroid)

    @property
    def centroid(self) -> tuple[float, float]:
        return self.detection.centroid

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return self.detection.bbox


class IOUTracker:
    """Frame-to-frame multi-object tracker matched by bounding-box IoU."""

    def __init__(self, max_age: int = 10, iou_threshold: float = 0.3, min_hits: int = 1) -> None:
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.min_hits = min_hits
        self.tracks: dict[int, Track] = {}
        self._next_id = 1

    def update(self, detections: list[Detection]) -> list[Track]:
        """Advance the tracker by one frame and return currently active tracks."""
        unmatched_detections = list(range(len(detections)))
        matches: list[tuple[int, int]] = []  # (track_id, detection_index)

        candidates: list[tuple[float, int, int]] = []
        for track_id, track in self.tracks.items():
            for det_idx in unmatched_detections:
                score = iou(track.bbox, detections[det_idx].bbox)
                if score >= self.iou_threshold:
                    candidates.append((score, track_id, det_idx))
        candidates.sort(key=lambda c: c[0], reverse=True)

        matched_track_ids: set[int] = set()
        matched_det_idxs: set[int] = set()
        for score, track_id, det_idx in candidates:
            if track_id in matched_track_ids or det_idx in matched_det_idxs:
                continue
            matches.append((track_id, det_idx))
            matched_track_ids.add(track_id)
            matched_det_idxs.add(det_idx)

        for track_id, det_idx in matches:
            track = self.tracks[track_id]
            track.detection = detections[det_idx]
            track.hits += 1
            track.age = 0
            track.frames_seen += 1
            track.history.append(track.detection.centroid)

        for track_id in list(self.tracks.keys()):
            if track_id not in matched_track_ids:
                track = self.tracks[track_id]
                track.age += 1
                if track.age > self.max_age:
                    del self.tracks[track_id]

        unmatched_det_idxs = [i for i in unmatched_detections if i not in matched_det_idxs]
        for det_idx in unmatched_det_idxs:
            new_id = self._next_id
            self._next_id += 1
            self.tracks[new_id] = Track(track_id=new_id, detection=detections[det_idx])

        return self.active_tracks()

    def active_tracks(self) -> list[Track]:
        return [t for t in self.tracks.values() if t.hits >= self.min_hits and t.age == 0]

    @property
    def total_tracks_spawned(self) -> int:
        """Total distinct track ids ever assigned (including now-dropped
        tracks) -- used by tasks reporting a final "unique objects seen"
        count after processing every frame."""
        return self._next_id - 1
