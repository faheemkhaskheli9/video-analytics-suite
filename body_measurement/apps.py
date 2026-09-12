from django.apps import AppConfig


class BodyMeasurementConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "body_measurement"

    def ready(self):
        from . import task  # noqa: F401 - import fires @register_task
