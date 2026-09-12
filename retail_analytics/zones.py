"""Zone definitions and point-in-polygon membership testing.

Ported from ``retail-video-analytics``'s ``zones/zone.py``. Zones are simple
polygons in frame pixel coordinates; ``kind`` tags a zone's role
(``"entrance"`` for entry/exit counting, ``"checkout_counter"`` for
cashier-presence tracking) so the event logic knows how to interpret
occupancy of that zone.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Zone:
    name: str
    polygon: tuple[tuple[float, float], ...]
    kind: str = "generic"

    def contains(self, point: tuple[float, float]) -> bool:
        """Ray-casting point-in-polygon test."""
        x, y = point
        n = len(self.polygon)
        if n < 3:
            return False
        inside = False
        x1, y1 = self.polygon[-1]
        for x2, y2 in self.polygon:
            if (y1 > y) != (y2 > y):
                x_intersect = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                if x < x_intersect:
                    inside = not inside
            x1, y1 = x2, y2
        return inside


class ZoneManager:
    """Holds all configured zones and reports which zones a point falls in."""

    def __init__(self, zones: list[Zone]) -> None:
        self.zones = zones

    def zones_containing(self, point: tuple[float, float]) -> list[Zone]:
        return [z for z in self.zones if z.contains(point)]

    def zones_of_kind(self, kind: str) -> list[Zone]:
        return [z for z in self.zones if z.kind == kind]


def default_zones(frame_width: int, frame_height: int) -> list[Zone]:
    """A plausible entrance + checkout-counter layout, scaled to the
    uploaded video's resolution -- there is no per-run zone editor in this
    MVP (see README Limitations), so every run uses this layout."""
    entrance_w = frame_width * 0.18
    checkout_w = frame_width * 0.22
    checkout_x0 = frame_width - checkout_w
    return [
        Zone(
            name="entrance",
            polygon=((0, 0), (entrance_w, 0), (entrance_w, frame_height), (0, frame_height)),
            kind="entrance",
        ),
        Zone(
            name="checkout_counter",
            polygon=(
                (checkout_x0, 0),
                (frame_width, 0),
                (frame_width, frame_height),
                (checkout_x0, frame_height),
            ),
            kind="checkout_counter",
        ),
    ]
