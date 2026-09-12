# Architecture

## Component overview

```text
Browser
  |
  v
Django view (jobs/views.py) -- validates upload against task.accepted_extensions
  |
  v
Job row created (status=queued), input_file saved under media/jobs/<id>/input/
  |
  v
run_video_job.delay(job_id)  -- Celery task (jobs/tasks.py)
  |
  v
video_core.registry.get_task(job.task_key)  -- string key, never a Python path
  |
  v
FeatureTask.run(input_path, tmp_output_dir, params)
  |         \
  |          -- video_core.io.sample_frames(...)          bounded-length frame sampling
  |          -- video_core.detection.build_detector(...)   HOG (default) or YOLO
  |          -- video_core.tracking.IOUTracker             greedy-IoU multi-object tracking
  |          -- video_core.pose.MediaPipePoseEstimator      pose landmarks (lazy import)
  |          -- video_core.visualize.*                      draw boxes/tracks/skeleton/heatmap
  v
tmp_output_dir fully written -> atomically renamed to output_dir (os.replace-equivalent)
  |
  v
Job.result = {metrics, visualization_paths, summary}; Job.status = done|failed
  |
  v
Browser polls /jobs/<id>/status.json until finished, then renders result
```

## Why a registry, not a dispatch table

Mirrors `medical-imaging-suite`'s pattern: `Job.task_key` is a plain string
(`"retail_analytics"`, `"pose_estimation"`, ...), never a Python import path
or callable reference. Each feature app's `apps.py::ready()` imports its
`task.py`, which fires `@register_task` and adds one `BaseVideoTask`
instance to `video_core.registry`'s in-memory dict. `jobs/tasks.py` is the
*only* place that ever calls `.run()` on a registered task -- there's no
second "run inline" code path to keep in sync. Adding a 9th feature is: new
Django app, subclass `BaseVideoTask`, decorate it, add the app to
`INSTALLED_APPS`. Nothing in `jobs` changes.

## Why frame sampling, not full-video decode

An uploaded clip can be minutes long. Every feature task calls
`video_core.io.sample_frames(path, frame_stride=settings.VIDEO_FRAME_STRIDE,
max_frames=settings.VIDEO_MAX_FRAMES)`, which yields frames one at a time
from `cv2.VideoCapture` (never loading the whole clip into memory) and stops
once `max_frames` sampled frames have been read, regardless of how much of
the source video that covers. This keeps a demo job's CPU time bounded on a
laptop-class worker; it also means metrics like "unique customers" or
"in/out counts" only reflect the sampled window, not the full upload --
every task's `summary` field says so.

## Why one shared `video_core`, not one copy per feature app

The nine source repos this suite consolidates had between two and five
near-duplicate implementations of: a `Detection` dataclass, a
`HOGPersonDetector`/`YoloDetector` pair behind a `Detector` interface, a
greedy-IoU tracker, and a MediaPipe pose wrapper. `video_core` is the single
canonical version of each, ported from whichever source repo's
implementation was most complete (`retail-video-analytics` for
detection/tracking/heatmap, `human-pose-estimation` /
`human-action-recognition` for the pose wrapper). Every feature app imports
from `video_core` instead of carrying its own copy, so a fix (e.g. the HOG
window-size guard) lands once for every feature that uses detection.

## Why an explicit, documented heuristic for action recognition

`human-action-recognition`'s source repo had shipped pose extraction
(Phase 1) but the action *classifier* (Phase 2) was still an open issue --
there was no trained model to port, unlike the medical-imaging-suite's
synthetic-data-trained U-Nets. Rather than leaving the feature unimplemented
or silently faking a "trained model," `action_recognition/classifier.py` is
a small, fully documented rule-based classifier over joint geometry (knee
angle) and frame-to-frame hip motion. `classroom_analysis` reuses the same
classifier rather than a second copy. Every result page says this
explicitly: it is a heuristic demo, not a trained action-recognition model.

## Why body measurement calibrates against an assumed height

`ai-body-measurement`'s source repo had shipped the human-detection gate
(exactly-one-person validation) but not the measurement math. A real
implementation would calibrate pixel-to-cm scale from either a reference
object in the photo or a user-entered height; this MVP has neither a
reference-object picker nor a height input field, so
`body_measurement/measurement.py` calibrates against a fixed assumed average
height (170cm) mapped to the detected person's bounding-box height in
pixels. This is stated on the result page, in the task's `summary`, and in
the README Limitations -- never presented as a real body-scan measurement.

## Atomic output, always

Every write that can be interrupted (encoding an annotated `.mp4`,
publishing a job's output directory) goes through the atomic-write pattern:
build in a temp location, then `os.replace`/`Path.rename` onto the final
path only after the write fully succeeds. `jobs/tasks.py` writes to
`.tmp_output_<task_id>/` and only renames it to `output/` after
`FeatureTask.run()` returns successfully; on any exception the tmp directory
is removed and `Job.status` becomes `failed` with the exception message
attached, so a worker crash mid-run never leaves the UI treating a
half-written result as done.
