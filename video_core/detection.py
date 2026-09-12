"""Shared detector interface + backends, used by every feature app.

Ported/consolidated from ``retail-video-analytics``'s
``detection/{base,hog_detector,yolo_detector}.py`` (the most complete of the
nine source repos this suite combines) so every feature app depends on one
``Detector`` contract instead of each repo's own near-duplicate of it.
``HOGPersonDetector`` (OpenCV's built-in HOG+SVM) is the default: it ships
inside opencv-python with no model download, so every task in this suite
runs offline out of the box. ``YoloDetector`` implements the same interface
for higher accuracy when ``ultralytics`` is installed and network access is
available to fetch weights (lazy-imported; never required just to import
this module).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

# COCO class id for "person" in the default Ultralytics models.
PERSON_CLASS_ID = 0


@dataclass(frozen=True)
class Detection:
    """A single detected object in one frame, in pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    label: str = "person"

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def centroid(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height


class Detector(ABC):
    """Abstract base class for a per-frame object detector."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Return detections for a single BGR frame (as produced by OpenCV)."""
        raise NotImplementedError


class HOGPersonDetector(Detector):
    """Person detector using ``cv2.HOGDescriptor``'s default people detector.

    Chosen as the default across every feature app because it requires no
    model download and no GPU -- the descriptor and its pre-trained linear
    SVM weights ship inside opencv-python itself.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.4,
        win_stride: tuple[int, int] = (8, 8),
        scale: float = 1.05,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.win_stride = win_stride
        self.scale = scale
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if frame is None or frame.size == 0:
            return []
        win_w, win_h = self._hog.winSize
        frame_h, frame_w = frame.shape[:2]
        if frame_h < win_h or frame_w < win_w:
            # Below the detector's own window size: real OpenCV builds treat
            # this as undefined behavior rather than "no detections" -- fail
            # safe by short-circuiting instead of calling detectMultiScale.
            return []
        rects, weights = self._hog.detectMultiScale(
            frame,
            winStride=self.win_stride,
            scale=self.scale,
        )
        detections: list[Detection] = []
        for (x, y, w, h), weight in zip(rects, weights):
            score = float(weight)
            # HOG's SVM decision function is unbounded; squash to (0, 1) so
            # it behaves like every other backend's confidence score.
            confidence = 1.0 / (1.0 + np.exp(-score))
            if confidence < self.confidence_threshold:
                continue
            detections.append(
                Detection(x1=float(x), y1=float(y), x2=float(x + w), y2=float(y + h), score=confidence)
            )
        return detections


class YoloDetector(Detector):
    """Wraps an Ultralytics YOLO model behind the shared :class:`Detector` API.

    ``model`` is a dependency-injection point used by tests (and by callers
    who already loaded a model) so this class never has to reach the network
    to be exercised.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.4,
        device: str = "cpu",
        classes: tuple[int, ...] = (PERSON_CLASS_ID,),
        model: Any | None = None,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.classes = classes
        self._model = model if model is not None else self._load_model(model_path)

    @staticmethod
    def _load_model(model_path: str) -> Any:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - exercised via mock in tests
            raise RuntimeError(
                "ultralytics is not installed. Install it (already in "
                "requirements.txt) or set DETECTOR_BACKEND=hog to use the "
                "offline default."
            ) from exc
        return YOLO(model_path)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self._model.predict(
            frame,
            device=self.device,
            classes=list(self.classes) if self.classes else None,
            conf=self.confidence_threshold,
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", [])
            for box in boxes:
                xyxy = _to_list(box.xyxy)
                conf = _to_scalar(box.conf)
                x1, y1, x2, y2 = xyxy[:4]
                detections.append(Detection(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2), score=float(conf)))
        return detections


def _to_list(value: Any) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], (list, tuple)):
        value = value[0]
    return list(value)


def _to_scalar(value: Any) -> float:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        value = value[0]
    return float(value)


def build_detector(backend: str, *, device: str = "cpu") -> Detector:
    """Config-driven factory used by every feature task's ``run()``."""
    if backend == "yolo":
        return YoloDetector(device=device)
    if backend == "hog":
        return HOGPersonDetector()
    raise ValueError(f"unknown detector backend {backend!r}; expected 'hog' or 'yolo'")
