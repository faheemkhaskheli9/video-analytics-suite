from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from django.conf import settings

from video_core.io import sample_frames, save_annotated_video
from video_core.pose import MediaPipePoseEstimator
from video_core.registry import BaseVideoTask, InputKind, TaskResult, VideoTaskError, register_task
from video_core.visualize import draw_skeleton

from .classifier import classify_frame, hip_centroid


@register_task
class ActionRecognitionTask(BaseVideoTask):
    key = "action_recognition"
    label = "Human Action Recognition"
    description = (
        "Pose extraction (MediaPipe) plus a rule-based classifier over "
        "joint geometry and hip motion, labeling each sampled frame as "
        "walking, sitting, presenting, or standing."
    )
    input_kind = InputKind.VIDEO
    accepted_extensions = (".mp4", ".avi", ".mov", ".mkv")

    def run(self, input_path: Path, output_dir: Path, params: dict[str, Any]) -> TaskResult:
        frames = list(sample_frames(input_path, frame_stride=settings.VIDEO_FRAME_STRIDE, max_frames=settings.VIDEO_MAX_FRAMES))
        if not frames:
            raise VideoTaskError("no frames could be read from the uploaded video")

        label_counts: Counter[str] = Counter()
        annotated_frames = []
        previous_hip = None
        with MediaPipePoseEstimator(static_image_mode=False) as estimator:
            for sample in frames:
                pose = estimator.predict(sample.image)
                label = classify_frame(pose, previous_hip, sample.image.shape[1], sample.image.shape[0])
                previous_hip = hip_centroid(pose)
                if label is not None:
                    label_counts[label] += 1
                annotated_frames.append(draw_skeleton(sample.image, pose, label=label or "no person detected"))

        video_path = save_annotated_video(annotated_frames, output_dir / "annotated.mp4", fps=6.0)

        frames_with_person = sum(label_counts.values())
        return TaskResult(
            metrics={
                "frames_processed": len(frames),
                "frames_with_person_detected": frames_with_person,
                "label_histogram": dict(label_counts),
                "dominant_action": label_counts.most_common(1)[0][0] if label_counts else None,
            },
            visualization_paths=[video_path.name],
            summary=(
                "Labels come from a rule-based heuristic over MediaPipe "
                "joint geometry (knee angle, wrist-vs-shoulder height, hip "
                "motion), not a trained action-recognition model -- see "
                "Limitations. Frames with no person detected are excluded "
                "from the histogram."
            ),
        )
