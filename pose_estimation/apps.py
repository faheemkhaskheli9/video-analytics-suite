from django.apps import AppConfig


class PoseEstimationConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "pose_estimation"

    def ready(self):
        from . import task  # noqa: F401 - import fires @register_task
