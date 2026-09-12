# Video Analytics Suite

> Computer Vision & Video Analytics portfolio project — independent open-source implementation.
> This is an original, from-scratch build. It is not affiliated with, and does not
> contain any code, prompts, data, or business logic from, any employer or client.

![status](https://img.shields.io/badge/status-mvp-yellow)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

## 1. Problem

Eight related video/CV-analytics tasks -- retail traffic analytics, object
tracking, line-crossing counting, traffic heatmaps, action recognition, pose
estimation, body measurement, and classroom teaching analysis -- each started
as its own scaffold-only repo with no shared UI and no way for someone to
just *try* the pipeline without cloning code and running scripts. This
project consolidates all eight into one Django web app: pick a task from a
dashboard, upload a video clip or image, get an annotated result back, with
real background-job processing behind it. It follows the same
one-suite-per-domain pattern as this portfolio's `medical-imaging-suite`.

## 2. Architecture

```text
Browser -> Django view -> Job row (queued) -> Celery task ->
  video_core (shared I/O/detection/tracking/pose/visualize) ->
  feature app's task.py -> Job.result (metrics + annotated video/PNG) ->
  Browser (polls until done)
```

See `docs/architecture.md` for the full component breakdown and design
rationale (registry pattern, why every feature samples frames instead of
decoding a whole clip, atomicity of job output).

## 3. Technology Stack

- Python, Django 5.2, Celery + Redis (background jobs), gunicorn
- OpenCV (`opencv-python-headless`) -- video/image I/O, the offline default
  HOG+SVM person detector, drawing, and heatmap rendering
- Ultralytics YOLO -- optional higher-accuracy detector backend (lazy
  import; needs network access to fetch weights on first use)
- MediaPipe -- pose landmark extraction (pose estimation, action
  recognition, body measurement); lazy-imported
- A from-scratch greedy-IoU multi-object tracker (no external tracking
  library dependency)
- Server-rendered templates, one token-based stylesheet, no JS build step
- sqlite by default; Postgres via `DATABASE_URL` for a real deployment

## 4. Feature List

- **Retail Video Analytics** -- detection + tracking + zone/event logic:
  customer counting, entry/exit counts, dwell time, cashier-absence alerts,
  and a foot-traffic heatmap.
- **Real-Time Object Tracking** -- YOLO/HOG detection + IoU tracking with
  persistent IDs, trajectory overlays, and an FPS benchmark.
- **Line-Crossing Counter** -- detection + tracking + a fixed counting line
  with in/out direction counts.
- **Traffic / Customer Heatmap** -- tracked-position density heatmap plus a
  per-track dwell-time summary.
- **Human Action Recognition** -- MediaPipe pose + a rule-based classifier
  (walking / sitting / presenting / standing) per sampled frame.
- **Human Pose Estimation** -- skeleton overlay, elbow/knee joint angles,
  and hip-displacement movement tracking across a clip.
- **Body Measurement from Image** -- single-person detection gate + pose
  landmarks + assumed-height scale calibration -> shoulder width and
  arm/leg length estimates in cm.
- **Classroom Teaching Analysis** -- most-central-person detection (teacher
  stand-in) + the same action classifier, aggregated into a per-lesson
  teaching-method time breakdown.
- Shared: user accounts, per-user job history, background job processing,
  atomic result writes, annotated-video/heatmap visualizations.

Every task runs on CPU with the offline HOG detector by default (no model
download, no GPU, no network access needed) -- each task page and result
summary is explicit about what's a heuristic/assumption vs. a measured
result. This is a pipeline demo, not a production analytics product.

## 5. Implementation Plan

1. Phase 1: Django scaffold + `video_core` (I/O, detection, tracking, pose,
   visualization, registry) + `jobs` app (model, Celery task,
   views/templates) -- **done**.
2. Phase 2: `retail_analytics` ported end-to-end from the most complete
   source repo (detection, tracking, zones, events, heatmap) -- **done**.
3. Phase 3: `object_tracking`, `line_crossing_counter`, `traffic_heatmap`
   built on the shared detector/tracker -- **done**.
4. Phase 4: `pose_estimation`, `action_recognition`, `body_measurement`,
   `classroom_analysis` built on the shared MediaPipe pose wrapper,
   including a documented rule-based action classifier where the source
   repo had only shipped pose extraction -- **done**.
5. Phase 5: `docker-compose.yml`, CI, docs, `PORTFOLIO_INDEX.md` update --
   **done**.
6. Future: swap the offline HOG default for YOLO in a hosted deployment;
   a real reference-object/user-entered-height calibration for body
   measurement; a trained (not rule-based) action classifier; per-run zone
   and counting-line editors.

## 6. Repository Structure

```text
video-analytics-suite/
├── README.md
├── LICENSE
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── .env.example
├── manage.py
├── config/                # settings, urls, celery app
├── video_core/            # shared I/O, detection, tracking, pose, visualize, registry
├── jobs/                  # Job model, Celery task, dashboard/job views
├── accounts/               # signup view
├── retail_analytics/       # feature app: zones.py + events.py + task.py
├── object_tracking/
├── line_crossing_counter/
├── traffic_heatmap/
├── action_recognition/     # + classifier.py (rule-based pose classifier)
├── pose_estimation/
├── body_measurement/       # + measurement.py (scale calibration math)
├── classroom_analysis/     # reuses action_recognition's classifier
├── templates/ + static/video/style.css
├── docker/                 # Dockerfile, docker-compose.yml
├── docs/
│   ├── architecture.md
│   └── evaluation.md
├── tests/  (per-app tests.py; this dir holds cross-cutting fixtures if any)
└── .github/workflows/ci.yml
```

## 7. Setup

```bash
git clone <this-repo-url>
cd video-analytics-suite
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env
python manage.py migrate
python manage.py createsuperuser   # optional, for /admin/
python manage.py runserver
```

Then open http://127.0.0.1:8000/, sign up, and pick a task.

## 8. Dataset

No public video-analytics dataset is bundled with this repo. Every task
runs real inference (detection, tracking, pose estimation) on whatever
video/image you upload -- there is no synthetic-data-trained model step like
`medical-imaging-suite`'s, because every detector/pose backend here is
either classical (OpenCV HOG+SVM, no training needed) or a pretrained
third-party model (Ultralytics YOLO COCO weights, MediaPipe Pose), not a
model trained in this repo. No proprietary, employer-owned, or
client-identifiable data is used in this project.

## 9. Training / Execution

No training step -- every detector/pose backend is either classical or
pretrained (see §8). The web flow runs inference directly: upload -> sampled
frames -> detector/tracker/pose -> annotated result. `settings.py`'s
`VIDEO_FRAME_STRIDE` / `VIDEO_MAX_FRAMES` bound how much of an uploaded clip
each job actually processes, keeping a demo job's CPU time bounded
regardless of upload length.

## 10. Evaluation

See `docs/evaluation.md` for per-feature metric definitions and a Result Log
to fill in as real clips are evaluated against.

## 11. Results

Every feature runs end-to-end on CPU: upload -> background job -> result
with metrics + an annotated video or heatmap PNG. See `docs/evaluation.md`
for what's measured vs. heuristic per task, and the honest caveat that
repeats across every task: the offline HOG detector and rule-based action
classifier are demo-grade, not production-accurate.

## 12. API

No REST API is exposed in this MVP -- everything is server-rendered Django
views (`jobs/urls.py`). A `Job` and its `result` JSON are visible at
`/jobs/<id>/`; `/jobs/<id>/status.json` is a small polling endpoint the job
page itself uses.

## 13. Docker

```bash
cd docker
docker compose up --build
```

Brings up `redis`, `web` (gunicorn, runs migrations on start), and `worker`
(Celery). See `docker-compose.yml` for environment variables (set
`DJANGO_SECRET_KEY` for anything beyond local testing).

## 14. Tests

```bash
pytest
```

Per-app unit tests: `video_core` (detection, tracking, heatmap, I/O
atomicity), `jobs` (job creation, atomic success/failure output, view
access control, upload validation), and a pipeline-math + task-smoke test
per feature app. Feature-app tests that need MediaPipe skip cleanly (rather
than fail) in an environment where it isn't installed.

## 15. Limitations

- This is a from-scratch, independent recreation built for portfolio
  purposes, consolidating eight smaller scaffold repos
  (`retail-video-analytics`, `realtime-object-tracking`,
  `yolo-line-crossing-counter`, `video-traffic-heatmap`,
  `human-action-recognition`, `human-pose-estimation`,
  `ai-body-measurement`, `classroom-teaching-analysis`), which remain as
  their own standalone repos, unmodified. `distributed-video-ai-platform`
  (Kafka/horizontal-scaling architecture demo) was deliberately **not**
  folded in here -- its focus is distributed-systems scaling, not a
  pick-a-task-and-upload UI, so it stays a separate repo.
- The offline `HOGPersonDetector` default is a classical, comparatively
  low-accuracy detector. `YoloDetector` (same interface) is the intended
  higher-accuracy option for a real deployment but needs `ultralytics` and
  a one-time network weights fetch.
- The tracker (`IOUTracker`) is a simplified greedy-IoU matcher with no
  Kalman filter or re-identification -- a dropped-and-regrown track gets a
  new ID, which can overcount unique objects/crossings under occlusion.
- `retail_analytics`'s zones and `line_crossing_counter`'s counting line are
  fixed layouts scaled to the uploaded video's resolution, not user-drawn
  per run.
- `action_recognition` / `classroom_analysis`'s action label comes from a
  documented rule-based heuristic over joint geometry and hip motion, not a
  trained action-recognition model. `classroom_analysis`'s "teacher" is the
  most horizontally-central detected person, not a trained
  teacher-vs-student classifier.
- `body_measurement`'s scale calibration assumes a fixed average height
  (170cm) rather than a reference object or user-entered height, so
  absolute cm values are only as accurate as that assumption.
- `traffic_heatmap` / `retail_analytics`'s dwell-time seconds assume a
  25fps source video rather than reading the file's real frame rate.
- sqlite is the default DB; for a real multi-container deployment, set
  `DATABASE_URL` to Postgres (sqlite has no real concurrent-writer story
  across the `web` and `worker` containers).

## 16. Future Work

- Swap the offline HOG default for YOLO in a hosted deployment and
  re-benchmark accuracy/FPS.
- Replace the rule-based action classifier with a trained one (e.g. a small
  LSTM/temporal-CNN over pose sequences) on a public action dataset.
- Per-run zone / counting-line editors instead of fixed layouts.
- A real reference-object or user-entered-height calibration path for body
  measurement.
- Track open items as GitHub Issues.

## 17. Disclosure

This repository is an **independent open-source recreation inspired by the kind of
production systems I have worked on professionally**. It contains no employer or
client source code, prompts, datasets, credentials, architecture diagrams, or
business logic. All code, data, and documentation here are original or built on
publicly available datasets and open-source tools.

---
_Last updated: 2026-09-12_
