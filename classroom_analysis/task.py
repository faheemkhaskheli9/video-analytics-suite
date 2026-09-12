"""Classroom teaching-method analysis, consolidated from the standalone
``classroom-teaching-analysis`` repo: teacher detection (largest/most-central
person box per frame, since there's no separate teacher-vs-student
classifier -- see Limitations) + the same rule-based pose classifier as
``action_recognition``, aggregated into a per-lesson teaching-method time
breakdown.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from django.conf import settings

from action_recognition.classifier import classify_frame, hip_centroid
from video_core.detection import Detection, build_detector
from video_core.io import sample_frames, save_annotated_video
from video_core.pose import MediaPipePoseEstimator
from video_core.registry import BaseVideoTask, InputKind, TaskResult, VideoTaskError, register_task
from video_core.visualize import draw_skeleton


def _most_central_detection(detections: list[Detection], width: int) -> Detection | None:
    """Picks the detection closest to horizontal frame-center as "the
    teacher" -- a stand-in for a real teacher-vs-student classifier, which
    this MVP does not have (see the task summary and README Limitations)."""
    if not detections:
        return None
    frame_center_x = width / 2.0
    return min(detections, key=lambda d: abs(d.centroid[0] - frame_center_x))


@register_task
class ClassroomAnalysisTask(BaseVideoTask):
    key = "classroom_analysis"
    label = "Classroom Teaching Analysis"
    description = (
        "Detects the most-central person per frame as a teacher stand-in, "
        "classifies their pose (walking / sitting / presenting / standing) "
        "with the same heuristic as Action Recognition, and aggregates a "
        "per-lesson time breakdown by teaching behavior."
    )
    input_kind = InputKind.VIDEO
    accepted_extensions = (".mp4", ".avi", ".mov", ".mkv")

    def run(self, input_path: Path, output_dir: Path, params: dict[str, Any]) -> TaskResult:
        detector = build_detector(settings.DETECTOR_BACKEND, device=settings.DETECTOR_DEVICE)

        frames = list(sample_frames(input_path, frame_stride=settings.VIDEO_FRAME_STRIDE, max_frames=settings.VIDEO_MAX_FRAMES))
        if not frames:
            raise VideoTaskError("no frames could be read from the uploaded video")

        seconds_per_sampled_frame = settings.VIDEO_FRAME_STRIDE / 25.0  # demo-scale; see summary
        label_seconds: Counter[str] = Counter()
        annotated_frames = []
        previous_hip = None
        frames_with_teacher = 0

        with MediaPipePoseEstimator(static_image_mode=False) as estimator:
            for sample in frames:
                detections = detector.detect(sample.image)
                teacher_box = _most_central_detection(detections, sample.image.shape[1])
                annotated = sample.image
                label = "no teacher detected"

                if teacher_box is not None:
                    frames_with_teacher += 1
                    pose = estimator.predict(sample.image)
                    action_label = classify_frame(pose, previous_hip, sample.image.shape[1], sample.image.shape[0])
                    previous_hip = hip_centroid(pose)
                    if action_label is not None:
                        label_seconds[action_label] += seconds_per_sampled_frame
                        label = action_label
                    annotated = draw_skeleton(sample.image, pose, label=label)

                annotated_frames.append(annotated)

        video_path = save_annotated_video(annotated_frames, output_dir / "annotated.mp4", fps=6.0)

        total_seconds = sum(label_seconds.values())
        breakdown_pct = {
            label: round(100 * seconds / total_seconds, 1) if total_seconds else 0.0
            for label, seconds in label_seconds.items()
        }

        return TaskResult(
            metrics={
                "frames_processed": len(frames),
                "frames_with_teacher_detected": frames_with_teacher,
                "teaching_method_seconds": {k: round(v, 1) for k, v in label_seconds.items()},
                "teaching_method_percent": breakdown_pct,
            },
            visualization_paths=[video_path.name],
            summary=(
                "\"Teacher\" is the most horizontally-central detected "
                "person per frame, not a trained teacher-vs-student "
                "classifier -- a multi-person classroom clip can mislabel "
                "a student as the teacher. Time estimates assume a 25fps "
                "source video. See Limitations."
            ),
        )
