import datetime as dt
import unittest

import numpy as np
import xarray as xr

from pygac_fdr.crop import crop_end


class CropTest(unittest.TestCase):
    def test_crop_end_same_day(self):
        """Test cropping if all observations belong to the same day."""
        ds = xr.Dataset(
            {"midnight_line": np.nan, "overlap_free_end": 123},
            attrs={"start_time": "20200101T1000Z", "end_time": "20200101T1200Z"},
        )

        # No date specified
        start_line, end_line = crop_end(ds, date=None)
        self.assertEqual(start_line, 0)
        self.assertEqual(end_line, 123)

        # Date specified matches start/end date
        start_line, end_line = crop_end(ds, date=dt.date(2020, 1, 1))
        self.assertEqual(start_line, 0)
        self.assertEqual(end_line, 123)

        # Date specified doesn't match
        start_line, end_line = crop_end(ds, date=dt.date(2020, 1, 2))
        self.assertEqual(start_line, None)
        self.assertEqual(end_line, None)

    def test_crop_end_previous_day(self):
        """Test cropping if observations cover previous and current day."""
        # 1. overlap_free_end > midnight_line
        ds = xr.Dataset(
            {"midnight_line": 100, "overlap_free_end": 123},
            attrs={"start_time": "20191231T2300Z", "end_time": "20200101T0100Z"},
        )
        start_line, end_line = crop_end(ds, date=dt.date(2020, 1, 1))
        self.assertEqual(start_line, 101)
        self.assertEqual(end_line, 123)

        # 2. overlap_free_end <= midnight_line (all observations within overlap)
        ds = xr.Dataset(
            {"midnight_line": 123, "overlap_free_end": 100},
            attrs={"start_time": "20191231T2300Z", "end_time": "20200101T0100Z"},
        )
        start_line, end_line = crop_end(ds, date=dt.date(2020, 1, 1))
        self.assertEqual(start_line, None)
        self.assertEqual(end_line, None)

    def test_crop_end_following_day(self):
        """Test cropping if observations cover current and following day."""
        # 1. overlap_free_end > midnight_line
        ds = xr.Dataset(
            {"midnight_line": 100, "overlap_free_end": 123},
            attrs={"start_time": "20200101T2300Z", "end_time": "20200102T0100Z"},
        )
        start_line, end_line = crop_end(ds, date=dt.date(2020, 1, 1))
        self.assertEqual(start_line, 0)
        self.assertEqual(end_line, 100)

        # 1. overlap_free_end > midnight_line
        ds = xr.Dataset(
            {"midnight_line": 123, "overlap_free_end": 100},
            attrs={"start_time": "20200101T2300Z", "end_time": "20200102T0100Z"},
        )
        start_line, end_line = crop_end(ds, date=dt.date(2020, 1, 1))
        self.assertEqual(start_line, 0)
        self.assertEqual(end_line, 100)
