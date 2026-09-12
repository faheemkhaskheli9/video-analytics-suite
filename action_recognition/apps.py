from django.apps import AppConfig


class ActionRecognitionConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "action_recognition"

    def ready(self):
        from . import task  # noqa: F401 - import fires @register_task
