"""Zone/event logic: dwell time, entry/exit counting, cashier presence.

Ported from ``retail-video-analytics``'s ``events/events.py``, adapted to
this suite's ``video_core.tracking.Track`` type. Turns a stream of per-frame
tracks plus zone definitions into the retail-relevant signals: customer
counting, entry/exit counts, dwell time, heatmap samples, cashier presence,
and cashier-absence duration.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from video_core.tracking import Track

from .zones import ZoneManager

ENTRANCE_KIND = "entrance"
CHECKOUT_KIND = "checkout_counter"


@dataclass
class DwellRecord:
    track_id: int
    zone_name: str
    seconds: float = 0.0
    entries: int = 0


@dataclass
class ZoneCounts:
    zone_name: str
    entries: int = 0
    exits: int = 0


@dataclass
class CashierAbsenceAlert:
    zone_name: str
    frame_index: int
    seconds: float


class EventEngine:
    """Stateful accumulator that turns tracked positions into retail events."""

    def __init__(self, zone_manager: ZoneManager, seconds_per_frame: float, cashier_absence_seconds: float = 5.0) -> None:
        self.zones = zone_manager
        self.seconds_per_frame = seconds_per_frame
        self.cashier_absence_seconds = cashier_absence_seconds

        self._occupancy: dict[str, set[int]] = {z.name: set() for z in self.zones.zones}
        self._dwell: dict[tuple[int, str], DwellRecord] = {}
        self._counts: dict[str, ZoneCounts] = {z.name: ZoneCounts(zone_name=z.name) for z in self.zones.zones}
        self._empty_streak: dict[str, int] = {z.name: 0 for z in self.zones.zones}
        self.cashier_alerts: list[CashierAbsenceAlert] = []
        self.max_concurrent_customers = 0
        self.heatmap_samples: list[tuple[float, float]] = []
        self._alerted_zones: set[str] = set()
        self._all_track_ids: set[int] = set()

    def process(self, frame_index: int, tracks: list[Track]) -> None:
        current_ids = {t.track_id for t in tracks}
        self._all_track_ids |= current_ids
        self.max_concurrent_customers = max(self.max_concurrent_customers, len(tracks))

        new_occupancy: dict[str, set[int]] = {name: set() for name in self._occupancy}
        for track in tracks:
            self.heatmap_samples.append(track.centroid)
            for zone in self.zones.zones_containing(track.centroid):
                new_occupancy[zone.name].add(track.track_id)
                key = (track.track_id, zone.name)
                record = self._dwell.setdefault(key, DwellRecord(track_id=track.track_id, zone_name=zone.name))
                if track.track_id not in self._occupancy[zone.name]:
                    record.entries += 1
                    if zone.kind == ENTRANCE_KIND:
                        self._counts[zone.name].entries += 1
                record.seconds += self.seconds_per_frame

        for zone_name, previous_ids in self._occupancy.items():
            left = previous_ids - new_occupancy[zone_name]
            if left:
                zone = next(z for z in self.zones.zones if z.name == zone_name)
                if zone.kind == ENTRANCE_KIND:
                    self._counts[zone_name].exits += len(left)

        self._occupancy = new_occupancy

        for zone in self.zones.zones_of_kind(CHECKOUT_KIND):
            occupied = len(self._occupancy[zone.name]) > 0
            if occupied:
                self._empty_streak[zone.name] = 0
                self._alerted_zones.discard(zone.name)
            else:
                self._empty_streak[zone.name] += 1
                absence_seconds = self._empty_streak[zone.name] * self.seconds_per_frame
                if absence_seconds >= self.cashier_absence_seconds and zone.name not in self._alerted_zones:
                    self._alerted_zones.add(zone.name)
                    self.cashier_alerts.append(
                        CashierAbsenceAlert(zone_name=zone.name, frame_index=frame_index, seconds=absence_seconds)
                    )

    def dwell_records(self) -> list[DwellRecord]:
        return list(self._dwell.values())

    def zone_counts(self) -> list[ZoneCounts]:
        return list(self._counts.values())

    def unique_customer_count(self) -> int:
        return len(self._all_track_ids)
