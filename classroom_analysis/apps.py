from django.apps import AppConfig


class ClassroomAnalysisConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "classroom_analysis"

    def ready(self):
        from . import task  # noqa: F401 - import fires @register_task
