from django.apps import AppConfig


class RetailAnalyticsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "retail_analytics"

    def ready(self):
        from . import task  # noqa: F401 - import fires @register_task
