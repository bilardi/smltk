import unittest
import numpy as np
import tests.variables as tv
from smltk.feature_engineering import Indicator


class TestIndicator(unittest.TestCase, Indicator):
    indicator = None
    timeseries = None
    timestamp = None

    def __init__(self, *args, **kwargs):
        self.indicator = Indicator()
        self.timeseries = self.get_wave()
        self.timestamp = range(0, 25)
        unittest.TestCase.__init__(self, *args, **kwargs)

    def get_wave(self):
        cycles = 4
        resolution = 25
        length = np.pi * 2 * cycles
        return np.sin(np.arange(0, length, length / resolution)) + 1

    def test_get_dc_events(self):
        events = self.indicator.get_dc_events(self.timeseries)
        self.assertEqual(events, tv.dc_events)

        indicator = Indicator({"timeseries": self.timeseries})
        self.assertEqual(events, indicator.get_dc_events())

        starts = self.indicator.get_dc_events_starts(events)
        directional_changes = set(events)
        for directional_change in directional_changes:
            self.assertEqual(
                starts[directional_change],
                tv.dc_events_starts[directional_change],
            )

        ends = self.indicator.get_dc_events_ends(events)
        directional_changes = set(events)
        for directional_change in directional_changes:
            self.assertEqual(
                ends[directional_change],
                tv.dc_events_ends[directional_change],
            )


if __name__ == "__main__":
    unittest.main()
