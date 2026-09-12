from __future__ import annotations

from pathlib import Path
from typing import Any

from video_core.detection import HOGPersonDetector
from video_core.io import load_image, save_image
from video_core.pose import MediaPipePoseEstimator
from video_core.registry import BaseVideoTask, InputKind, TaskResult, VideoTaskError, register_task
from video_core.visualize import draw_skeleton

from .measurement import ASSUMED_HEIGHT_CM, compute_measurements, pixels_per_cm


class NoPersonDetectedError(VideoTaskError):
    pass


class MultiplePeopleDetectedError(VideoTaskError):
    pass


@register_task
class BodyMeasurementTask(BaseVideoTask):
    key = "body_measurement"
    label = "Body Measurement from Image"
    description = (
        "Detects exactly one person in a photo, extracts MediaPipe pose "
        "landmarks, and estimates shoulder width and arm/leg length in cm "
        "by calibrating pixel scale against an assumed average body height."
    )
    input_kind = InputKind.IMAGE
    accepted_extensions = (".png", ".jpg", ".jpeg")

    def run(self, input_path: Path, output_dir: Path, params: dict[str, Any]) -> TaskResult:
        image = load_image(input_path)
        height, width = image.shape[:2]

        detector = HOGPersonDetector()
        detections = detector.detect(image)
        if len(detections) == 0:
            raise NoPersonDetectedError(
                "no person detected in the image -- try a clearer full-body photo with good contrast"
            )
        if len(detections) > 1:
            raise MultiplePeopleDetectedError(
                f"expected exactly one person, found {len(detections)} -- crop the photo to one person"
            )
        person = detections[0]

        with MediaPipePoseEstimator(static_image_mode=True) as estimator:
            pose = estimator.predict(image)
        if not pose.detected:
            raise VideoTaskError("a person was detected by the box detector, but pose landmarks could not be extracted")

        px_per_cm = pixels_per_cm(person.height, ASSUMED_HEIGHT_CM)
        measurements = compute_measurements(pose, width, height, px_per_cm)

        label = ", ".join(f"{k.replace('_cm', '')}: {v}cm" for k, v in measurements.items())
        annotated = draw_skeleton(image, pose, label=label)
        image_path = save_image(annotated, output_dir / "measurements.png")

        return TaskResult(
            metrics={**measurements, "assumed_height_cm": ASSUMED_HEIGHT_CM, "detection_confidence": round(person.score, 3)},
            visualization_paths=[image_path.name],
            summary=(
                f"Scale is calibrated from an *assumed* average height of "
                f"{ASSUMED_HEIGHT_CM}cm mapped to the detected person's "
                "pixel bounding-box height -- there is no reference object "
                "or user-entered height in this MVP, so absolute cm values "
                "are only as accurate as that assumption. See Limitations."
            ),
        )
