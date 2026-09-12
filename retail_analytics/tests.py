from __future__ import annotations

from django.test import SimpleTestCase

from video_core.detection import Detection
from video_core.tracking import IOUTracker

from .events import EventEngine
from .zones import Zone, ZoneManager, default_zones


class ZoneTests(SimpleTestCase):
    def test_point_inside_rectangle_zone(self):
        zone = Zone(name="a", polygon=((0, 0), (10, 0), (10, 10), (0, 10)))
        self.assertTrue(zone.contains((5, 5)))
        self.assertFalse(zone.contains((15, 5)))

    def test_default_zones_scale_to_frame_size(self):
        zones = default_zones(1000, 500)
        entrance = next(z for z in zones if z.kind == "entrance")
        checkout = next(z for z in zones if z.kind == "checkout_counter")
        self.assertTrue(entrance.contains((10, 250)))
        self.assertTrue(checkout.contains((990, 250)))
        self.assertFalse(entrance.contains((500, 250)))


class EventEngineTests(SimpleTestCase):
    def test_entry_and_exit_counted_for_entrance_zone(self):
        zone = Zone(name="entrance", polygon=((0, 0), (100, 0), (100, 100), (0, 100)), kind="entrance")
        engine = EventEngine(ZoneManager([zone]), seconds_per_frame=0.1)
        tracker = IOUTracker()

        # Frame 1: a person enters the zone.
        tracks = tracker.update([Detection(10, 10, 30, 30, 0.9)])
        engine.process(0, tracks)
        # Frame 2: they leave the zone (moved far away).
        tracks = tracker.update([Detection(500, 500, 520, 520, 0.9)])
        engine.process(1, tracks)

        counts = {zc.zone_name: zc for zc in engine.zone_counts()}
        self.assertEqual(counts["entrance"].entries, 1)
        self.assertEqual(counts["entrance"].exits, 1)

    def test_cashier_absence_alert_fires_after_threshold(self):
        zone = Zone(name="checkout", polygon=((0, 0), (100, 0), (100, 100), (0, 100)), kind="checkout_counter")
        engine = EventEngine(ZoneManager([zone]), seconds_per_frame=1.0, cashier_absence_seconds=2.0)
        for frame_index in range(3):
            engine.process(frame_index, [])  # never occupied
        self.assertEqual(len(engine.cashier_alerts), 1)
        self.assertGreaterEqual(engine.cashier_alerts[0].seconds, 2.0)

    def test_max_concurrent_customers_tracks_peak(self):
        zone = Zone(name="z", polygon=((0, 0), (1000, 0), (1000, 1000), (0, 1000)))
        engine = EventEngine(ZoneManager([zone]), seconds_per_frame=0.1)
        tracker = IOUTracker()
        tracks = tracker.update([Detection(0, 0, 10, 10, 0.9), Detection(200, 200, 210, 210, 0.9)])
        engine.process(0, tracks)
        tracks = tracker.update([Detection(0, 0, 10, 10, 0.9)])
        engine.process(1, tracks)
        self.assertEqual(engine.max_concurrent_customers, 2)
        self.assertEqual(engine.unique_customer_count(), 2)
