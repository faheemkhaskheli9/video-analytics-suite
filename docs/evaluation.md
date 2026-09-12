# Evaluation

Every feature below is an inference-only demo (no dataset ships with this
repo -- see README §8). This document tracks what's *measured* per run
(reported in `Job.result.metrics`) vs. what's a stated assumption/heuristic,
and a Result Log to fill in as real clips are run against each task.

## Per-feature metrics

| Feature | What's measured | What's assumed / heuristic |
|---|---|---|
| Retail Video Analytics | unique customers, max concurrent, zone entries/exits, average dwell seconds, cashier-absence alert count, heatmap sample count | entrance/checkout zone layout is a fixed shape scaled to the video's resolution, not drawn per run; seconds-per-frame assumes 25fps |
| Object Tracking | frames processed, total tracks spawned, tracks still active at end, average track length (frames), detect+track FPS | a dropped-and-regrown track gets a new id (no re-identification), so "total tracks spawned" over-counts under occlusion |
| Line-Crossing Counter | in count, out count, net count | the counting line is fixed at the frame's vertical midline, not user-drawn |
| Traffic / Customer Heatmap | position sample count, peak cell density, unique tracks, top-5 per-track dwell seconds | dwell seconds assume 25fps source video |
| Human Action Recognition | frames with person detected, per-label frame counts, dominant action | labels come from a rule-based heuristic (knee angle, wrist-vs-shoulder height, hip motion), not a trained classifier |
| Human Pose Estimation | frames with person detected, average elbow/knee joint angles (degrees), total hip movement (px) | movement is a pixel-space sum, not a calibrated real-world distance |
| Body Measurement | shoulder width, arm length (L/R), leg length (L/R), all in cm; detection confidence | scale is calibrated against an *assumed* 170cm average height, not a reference object or user-entered height |
| Classroom Teaching Analysis | frames with teacher detected, per-behavior seconds and percent-of-lesson | "teacher" = most horizontally-central detected person, not a trained teacher-vs-student classifier; seconds assume 25fps |

## How to reproduce a result

1. `python manage.py runserver`, sign up, pick a task from the dashboard.
2. Upload a short clip (a few seconds is enough to exercise every metric --
   `VIDEO_MAX_FRAMES` bounds how much of a longer upload is actually
   processed).
3. The job page shows `Job.result.metrics` and the annotated
   video/heatmap once the job finishes.

## Result Log

Fill in as real clips are run. `frames_processed` reflects
`VIDEO_FRAME_STRIDE`/`VIDEO_MAX_FRAMES` at run time, not the source video's
actual frame count.

| Date | Feature | Input | Key metric(s) | Notes |
|---|---|---|---|---|
| _(none yet)_ | | | | |
