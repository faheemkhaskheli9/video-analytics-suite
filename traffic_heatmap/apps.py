from django.apps import AppConfig


class TrafficHeatmapConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "traffic_heatmap"

    def ready(self):
        from . import task  # noqa: F401 - import fires @register_task
