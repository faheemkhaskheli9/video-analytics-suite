from __future__ import annotations

import tempfile
from pathlib import Path

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse

from video_core import registry as video_registry
from video_core.registry import BaseVideoTask, InputKind, TaskResult, VideoTaskError

from .models import Job
from .tasks import run_video_job

User = get_user_model()


class _OkTask(BaseVideoTask):
    key = "test_ok_task"
    label = "Test OK Task"
    description = "A fake task that always succeeds, for jobs-app tests."
    input_kind = InputKind.IMAGE
    accepted_extensions = (".png", ".jpg")

    def run(self, input_path: Path, output_dir: Path, params: dict) -> TaskResult:
        (output_dir / "result.png").write_bytes(b"fake-png-bytes")
        return TaskResult(metrics={"ok": True}, visualization_paths=["result.png"], summary="done")


class _FailTask(BaseVideoTask):
    key = "test_fail_task"
    label = "Test Fail Task"
    description = "A fake task that always raises, for jobs-app tests."
    input_kind = InputKind.IMAGE
    accepted_extensions = (".png",)

    def run(self, input_path: Path, output_dir: Path, params: dict) -> TaskResult:
        (output_dir / "partial.txt").write_text("should never be visible")
        raise VideoTaskError("boom")


def _register(task_cls):
    instance = task_cls()
    video_registry._REGISTRY[instance.key] = instance
    return instance


def _unregister(key: str):
    video_registry._REGISTRY.pop(key, None)


class JobPipelineTests(TestCase):
    def setUp(self):
        _register(_OkTask)
        _register(_FailTask)
        self.addCleanup(_unregister, "test_ok_task")
        self.addCleanup(_unregister, "test_fail_task")
        # An isolated MEDIA_ROOT per test run -- writing job output under the
        # project's real media/ dir would leave leftover files across runs
        # (Job ids restart at 1 each test run, so a stale media/jobs/1/
        # from an earlier run could collide with this run's assertions).
        media_root = tempfile.TemporaryDirectory()
        self.addCleanup(media_root.cleanup)
        override = override_settings(MEDIA_ROOT=media_root.name)
        override.enable()
        self.addCleanup(override.disable)
        self.user = User.objects.create_user(username="alice", password="pw12345!")

    def _make_job(self, task_key: str) -> Job:
        job = Job.objects.create(owner=self.user, task_key=task_key, params={})
        job.input_file = SimpleUploadedFile("input.png", b"fake-bytes")
        job.save(update_fields=["input_file"])
        return job

    def test_successful_job_publishes_output_atomically(self):
        job = self._make_job("test_ok_task")
        run_video_job(job.pk)
        job.refresh_from_db()
        self.assertEqual(job.status, Job.Status.DONE)
        self.assertEqual(job.result["metrics"], {"ok": True})
        self.assertTrue(job.output_dir.endswith("output"))

    def test_failed_job_records_error_and_leaves_no_partial_output(self):
        job = self._make_job("test_fail_task")
        run_video_job(job.pk)
        job.refresh_from_db()
        self.assertEqual(job.status, Job.Status.FAILED)
        self.assertIn("boom", job.error)
        self.assertEqual(job.output_dir, "")
        # The failing task wrote a file before raising -- it must not survive
        # (tmp output dir is removed on failure, never published/renamed).
        from django.conf import settings

        job_dir = Path(settings.MEDIA_ROOT) / "jobs" / str(job.pk)
        leftover_dirs = list(job_dir.glob("*output*")) if job_dir.exists() else []
        self.assertEqual(leftover_dirs, [])


class JobViewTests(TestCase):
    def setUp(self):
        _register(_OkTask)
        self.addCleanup(_unregister, "test_ok_task")
        media_root = tempfile.TemporaryDirectory()
        self.addCleanup(media_root.cleanup)
        override = override_settings(MEDIA_ROOT=media_root.name)
        override.enable()
        self.addCleanup(override.disable)
        self.user = User.objects.create_user(username="alice", password="pw12345!")
        self.other = User.objects.create_user(username="bob", password="pw12345!")

    def test_dashboard_lists_registered_tasks(self):
        response = self.client.get(reverse("jobs:dashboard"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Test OK Task")

    def test_unregistered_task_detail_is_404(self):
        response = self.client.get(reverse("jobs:task_detail", args=["nope"]))
        self.assertEqual(response.status_code, 404)

    def test_anonymous_submit_redirects_to_login(self):
        response = self.client.post(
            reverse("jobs:task_detail", args=["test_ok_task"]),
            {"input_file": SimpleUploadedFile("in.png", b"x")},
        )
        self.assertEqual(response.status_code, 302)
        self.assertIn("/accounts/login/", response.url)

    def test_wrong_extension_is_rejected_by_form_validation(self):
        self.client.login(username="alice", password="pw12345!")
        response = self.client.post(
            reverse("jobs:task_detail", args=["test_ok_task"]),
            {"input_file": SimpleUploadedFile("in.mp4", b"x")},
        )
        self.assertEqual(response.status_code, 200)  # re-rendered with form errors
        self.assertEqual(Job.objects.count(), 0)

    @override_settings(CELERY_TASK_ALWAYS_EAGER=True, CELERY_TASK_EAGER_PROPAGATES=True)
    def test_authenticated_submit_creates_and_runs_job(self):
        self.client.login(username="alice", password="pw12345!")
        response = self.client.post(
            reverse("jobs:task_detail", args=["test_ok_task"]),
            {"input_file": SimpleUploadedFile("in.png", b"x")},
        )
        self.assertEqual(response.status_code, 302)
        job = Job.objects.get()
        self.assertEqual(job.status, Job.Status.DONE)

    def test_job_detail_is_owner_or_staff_only(self):
        job = Job.objects.create(owner=self.user, task_key="test_ok_task")
        self.client.login(username="bob", password="pw12345!")
        response = self.client.get(job.get_absolute_url())
        self.assertEqual(response.status_code, 404)
