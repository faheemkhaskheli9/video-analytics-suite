"""Shared video/image I/O for every feature app.

Every task samples frames from an uploaded clip rather than decoding it
whole into memory (a long upload must not OOM a worker), and every write
that can be interrupted goes to a temp path + ``os.replace`` (portfolio
rule 1) so a crash mid-encode never leaves a truncated result file behind.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .registry import VideoTaskError


class VideoReadError(VideoTaskError):
    """The uploaded file could not be opened/decoded as a video."""


@dataclass(frozen=True)
class SampledFrame:
    index: int          # sequence number among *sampled* frames
    source_index: int   # frame number in the original video
    timestamp_s: float
    image: np.ndarray   # HxWx3 BGR


@dataclass(frozen=True)
class VideoMeta:
    width: int
    height: int
    fps: float
    frame_count: int


def probe_video(path: str | Path) -> VideoMeta:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise VideoReadError(f"could not open video (unreadable/unsupported format): {path}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        return VideoMeta(width=width, height=height, fps=fps or 25.0, frame_count=frame_count)
    finally:
        cap.release()


def sample_frames(
    path: str | Path,
    *,
    frame_stride: int = 3,
    max_frames: int = 150,
) -> Iterator[SampledFrame]:
    """Yield up to ``max_frames`` frames, one every ``frame_stride`` source
    frames, so a long upload still processes in bounded time on CPU."""
    path = Path(path)
    if not path.is_file():
        raise VideoReadError(f"video file not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise VideoReadError(f"could not open video (unreadable/unsupported format): {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    try:
        source_index = 0
        sampled_index = 0
        while sampled_index < max_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if source_index % frame_stride == 0:
                yield SampledFrame(
                    index=sampled_index,
                    source_index=source_index,
                    timestamp_s=source_index / fps,
                    image=frame,
                )
                sampled_index += 1
            source_index += 1
    finally:
        cap.release()


def load_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise VideoReadError(f"image file not found: {path}")
    image = cv2.imread(str(path))
    if image is None:
        raise VideoReadError(f"could not decode image (unsupported/corrupt file): {path}")
    return image


def _atomic_write_bytes(target: Path, data: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def save_image(image: np.ndarray, out_path: Path) -> Path:
    ok, buf = cv2.imencode(".png", image)
    if not ok:  # pragma: no cover - cv2 PNG encode does not realistically fail here
        raise VideoTaskError("failed to PNG-encode image")
    _atomic_write_bytes(out_path, buf.tobytes())
    return out_path


def save_annotated_video(frames: list[np.ndarray], out_path: Path, *, fps: float = 8.0) -> Path:
    """Write ``frames`` (already annotated, BGR) as an MP4 the browser can
    play. Encoded to a temp path first, then renamed atomically so a worker
    crash mid-encode never leaves a truncated video the UI treats as done."""
    if not frames:
        raise VideoTaskError("no frames to write")
    height, width = frames[0].shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f".{out_path.name}.{os.getpid()}.tmp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_path), fourcc, fps, (width, height))
    try:
        if not writer.isOpened():
            raise VideoTaskError("could not open video encoder (mp4v codec unavailable)")
        for frame in frames:
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
    finally:
        writer.release()
    os.replace(tmp_path, out_path)
    return out_path
