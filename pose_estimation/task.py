"""Human pose estimation toolkit, consolidated from the standalone
``human-pose-estimation`` repo: MediaPipe pose landmarks, a skeleton
overlay, elbow/knee joint-angle computation, and movement tracking (hip
displacement) across a clip's sampled frames.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from django.conf import settings

from video_core.io import sample_frames, save_annotated_video
from video_core.pose import MediaPipePoseEstimator, joint_angle_degrees
from video_core.registry import BaseVideoTask, InputKind, TaskResult, VideoTaskError, register_task
from video_core.visualize import draw_skeleton


@register_task
class PoseEstimationTask(BaseVideoTask):
    key = "pose_estimation"
    label = "Human Pose Estimation"
    description = (
        "MediaPipe pose landmarks with a skeleton overlay, elbow/knee joint "
        "angles per sampled frame, and hip-displacement movement tracking "
        "across the clip."
    )
    input_kind = InputKind.VIDEO
    accepted_extensions = (".mp4", ".avi", ".mov", ".mkv")

    def run(self, input_path: Path, output_dir: Path, params: dict[str, Any]) -> TaskResult:
        frames = list(sample_frames(input_path, frame_stride=settings.VIDEO_FRAME_STRIDE, max_frames=settings.VIDEO_MAX_FRAMES))
        if not frames:
            raise VideoTaskError("no frames could be read from the uploaded video")

        annotated_frames = []
        angle_sums = {"left_elbow": 0.0, "right_elbow": 0.0, "left_knee": 0.0, "right_knee": 0.0}
        angle_counts = {k: 0 for k in angle_sums}
        total_movement_px = 0.0
        previous_hip_px: tuple[float, float] | None = None
        frames_with_person = 0

        with MediaPipePoseEstimator(static_image_mode=False) as estimator:
            for sample in frames:
                height, width = sample.image.shape[:2]
                pose = estimator.predict(sample.image)
                label = ""
                if pose.detected:
                    frames_with_person += 1
                    joints = {
                        "left_elbow": ("left_shoulder", "left_elbow", "left_wrist"),
                        "right_elbow": ("right_shoulder", "right_elbow", "right_wrist"),
                        "left_knee": ("left_hip", "left_knee", "left_ankle"),
                        "right_knee": ("right_hip", "right_knee", "right_ankle"),
                    }
                    angles = {}
                    for name, (a, b, c) in joints.items():
                        angle = joint_angle_degrees(pose, a, b, c, width, height)
                        if angle is not None:
                            angle_sums[name] += angle
                            angle_counts[name] += 1
                            angles[name] = round(angle, 1)
                    label = ", ".join(f"{k}: {v:.0f}deg" for k, v in angles.items())

                    hip_px = pose.pixel_xy("left_hip", width, height)
                    if hip_px and previous_hip_px:
                        dx, dy = hip_px[0] - previous_hip_px[0], hip_px[1] - previous_hip_px[1]
                        total_movement_px += (dx**2 + dy**2) ** 0.5
                    previous_hip_px = hip_px

                annotated_frames.append(draw_skeleton(sample.image, pose, label=label))

        video_path = save_annotated_video(annotated_frames, output_dir / "annotated.mp4", fps=6.0)

        average_angles = {
            name: round(angle_sums[name] / angle_counts[name], 1) for name in angle_sums if angle_counts[name] > 0
        }

        return TaskResult(
            metrics={
                "frames_processed": len(frames),
                "frames_with_person_detected": frames_with_person,
                "average_joint_angles_degrees": average_angles,
                "total_hip_movement_px": round(total_movement_px, 1),
            },
            visualization_paths=[video_path.name],
            summary=(
                "Joint angles are computed per sampled frame from MediaPipe "
                "landmarks (shoulder-elbow-wrist, hip-knee-ankle). Movement "
                "is a pixel-space hip displacement sum across sampled "
                "frames, not a calibrated real-world distance -- see "
                "Limitations."
            ),
        )
