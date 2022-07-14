#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2020 pygac-fdr developers
#
# This file is part of pygac-fdr.
#
# pygac-fdr is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# pygac-fdr is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# pygac-fdr. If not, see <http://www.gnu.org/licenses/>.

from unittest import mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pygac_fdr.metadata import (
    MetadataCollector,
    MetadataEnhancer,
    QualityFlags,
    read_metadata_from_database,
    save_metadata_to_database,
)


def open_dataset_patched(filename):
    times = {
        "file3": [
            np.datetime64("2009-07-01 00:00"),
            np.datetime64("2009-07-01 00:30"),
            np.datetime64("2009-07-01 00:45"),
            np.datetime64("2009-07-01 00:55"),
            np.datetime64("2009-07-01 01:00"),
        ],
        "file4": [
            np.datetime64("2009-07-01 00:50"),
            np.datetime64("2009-07-01 01:00"),
            np.datetime64("2009-07-01 01:30"),
            np.datetime64("2009-07-01 01:49"),
            np.datetime64("2009-07-01 02:00"),
        ],
        "file5": [
            np.datetime64("2009-07-01 01:50"),
            np.datetime64("2009-07-01 02:01"),
            np.datetime64("2009-07-01 02:30"),
            np.datetime64("2009-07-01 02:51"),
            np.datetime64("2009-07-01 03:00"),
        ],
        "file9": [
            np.datetime64("2009-07-01 02:50"),
            np.datetime64("2009-07-01 03:00"),
            np.datetime64("2009-07-01 03:15"),
            np.datetime64("2009-07-01 03:45"),
            np.datetime64("2009-07-01 04:00"),
        ],
    }
    times = dict(
        [(fname, xr.Dataset({"acq_time": time})) for fname, time in times.items()]
    )
    return times.get(filename, xr.Dataset({"acq_time": [0]}))


class TestMetadataCollector:
    def test_get_midnight_line(self):
        acq_time = xr.DataArray(
            [
                np.datetime64("2009-12-31 23:58:00"),
                np.datetime64("2009-12-31 23:59:00"),
                np.datetime64("2010-01-01 00:00:01"),
                np.datetime64("2010-01-01 00:01:00"),
            ]
        )
        collector = MetadataCollector()
        midn_line = collector._get_midnight_line(acq_time)
        assert midn_line == 1

        # No date switch
        acq_time = xr.DataArray(
            [
                np.datetime64("2010-01-01 00:00:01"),
                np.datetime64("2010-01-01 00:01:00"),
                np.datetime64("2010-01-01 00:02:00"),
            ]
        )
        np.testing.assert_equal(collector._get_midnight_line(acq_time), np.nan)

    def test_get_equator_crossings(self):
        collector = MetadataCollector()

        # No equator crossing
        ds = xr.Dataset(
            coords={
                "latitude": (
                    ("y", "x"),
                    [[999, 5.0, 999], [999, 0.1, 999], [999, -5.0, 999]],
                ),
                "longitude": (
                    ("y", "x"),
                    [[999, 1.0, 999], [999, 2.0, 999], [999, 3.0, 999]],
                ),
                "acq_time": ("y", np.arange(3).astype("datetime64[s]")),
            }
        )
        lons, times = collector._get_equator_crossings(ds)
        np.testing.assert_equal(lons, [np.nan, np.nan])
        assert np.all(np.isnat(times))

        # One equator crossing
        ds = xr.Dataset(
            coords={
                "latitude": (
                    ("y", "x"),
                    [
                        [999, 5.0, 999],
                        [999, 0.1, 999],
                        [999, -5.0, 999],
                        [999, 0.1, 999],
                        [999, 5.0, 999],
                    ],
                ),
                "longitude": (
                    ("y", "x"),
                    [
                        [999, 1.0, 999],
                        [999, 2.0, 999],
                        [999, 3.0, 999],
                        [999, 4.0, 999],
                        [999, 5.0, 999],
                    ],
                ),
                "acq_time": ("y", np.arange(5).astype("datetime64[s]")),
            }
        )
        ds["acq_time"].attrs["coords"] = "latitude longitude"
        lons, times = collector._get_equator_crossings(ds)
        np.testing.assert_equal(lons, [3, np.nan])
        np.testing.assert_equal(
            times, [np.datetime64("1970-01-01 00:00:02"), np.datetime64("NaT")]
        )

        # More than two equator crossings
        ds = xr.Dataset(
            coords={
                "latitude": (
                    ("y", "x"),
                    [
                        [999, -1, 999],
                        [999, 1, 999],
                        [999, -1, 999],
                        [999, 1, 999],
                        [999, -1, 999],
                        [999, 1, 999],
                    ],
                ),
                "longitude": (
                    ("y", "x"),
                    [
                        [999, 1.0, 999],
                        [999, 2.0, 999],
                        [999, 3.0, 999],
                        [999, 4.0, 999],
                        [999, 5.0, 999],
                        [999, 6.0, 999],
                    ],
                ),
                "acq_time": ("y", np.arange(6).astype("datetime64[s]")),
            }
        )
        ds["acq_time"].attrs["coords"] = "latitude longitude"
        collector = MetadataCollector()
        lons, times = collector._get_equator_crossings(ds)
        np.testing.assert_equal(lons, [1, 3])
        np.testing.assert_equal(
            times,
            [
                np.datetime64("1970-01-01 00:00:00"),
                np.datetime64("1970-01-01 00:00:02"),
            ],
        )


class TestMetadataEnhancer:
    def get_mda(self, multi_platform=False, reverse=False):
        mda = [
            {
                "platform": "NOAA-16",
                "start_time": np.datetime64("2020-01-01 12:00"),
                "end_time": np.datetime64("2020-01-01 13:00"),
                "along_track": 12000,
                "filename": "file1",
                "midnight_scanline": 1234,
                "overlap_free_start": np.nan,
                "overlap_free_start_exp": np.nan,
                "overlap_free_end": np.nan,
                "overlap_free_end_exp": np.nan,
                "global_quality_flag": QualityFlags.OK,
                "global_quality_flag_exp": QualityFlags.INVALID_TIMESTAMP,
            },
            {
                "platform": "NOAA-16",
                "start_time": np.datetime64("2009-06-30"),
                "end_time": np.datetime64("2049-01-01"),
                "along_track": 12000,
                "filename": "file2",
                "midnight_scanline": 1234,
                "overlap_free_start": np.nan,
                "overlap_free_start_exp": np.nan,
                "overlap_free_end": np.nan,
                "overlap_free_end_exp": np.nan,
                "global_quality_flag": QualityFlags.OK,
                "global_quality_flag_exp": QualityFlags.TOO_LONG,
            },
            {
                "platform": "NOAA-16",
                "start_time": np.datetime64("2009-07-01 00:00"),
                "end_time": np.datetime64("2009-07-01 01:00"),
                "along_track": 12000,
                "filename": "file3",
                "midnight_scanline": 1234,
                "overlap_free_start": np.nan,
                "overlap_free_start_exp": 0,
                "overlap_free_end": np.nan,
                "overlap_free_end_exp": 2,
                "global_quality_flag": QualityFlags.OK,
                "global_quality_flag_exp": QualityFlags.OK,
            },
            {
                "platform": "NOAA-16",
                "start_time": np.datetime64("2009-07-01 00:50"),
                "end_time": np.datetime64("2009-07-01 02:00"),
                "along_track": 12000,
                "filename": "file4",
                "midnight_scanline": 1234,
                "overlap_free_start": np.nan,
                "overlap_free_start_exp": 2,
                "overlap_free_end": np.nan,
                "overlap_free_end_exp": 3,
                "global_quality_flag": QualityFlags.OK,
                "global_quality_flag_exp": QualityFlags.OK,
            },
            {
                "platform": "NOAA-16",
                "start_time": np.datetime64("2009-07-01 01:50"),
                "end_time": np.datetime64("2009-07-01 03:00"),
                "along_track": 12000,
                "filename": "file5",
                "midnight_scanline": 1234,
                "overlap_free_start": np.nan,
                "overlap_free_start_exp": 1,
                "overlap_free_end": np.nan,
                "overlap_free_end_exp": 2,
                "global_quality_flag": QualityFlags.OK,
                "global_quality_flag_exp": QualityFlags.OK,
            },
            {
                "platform": "NOAA-16",
                "start_time": np.datetime64("2009-07-01 02:15"),
                "end_time": np.datetime64("2009-07-01 02:30"),
                "along_track": 12000,
                "filename": "file6",
                "midnight_scanline": 1234,
                "overlap_free_start": np.nan,
                "overlap_free_start_exp": np.nan,
                "overlap_free_end": np.nan,
                "overlap_free_end_exp": np.nan,
                "global_quality_flag": QualityFlags.OK,
                "global_quality_flag_exp": QualityFlags.REDUNDANT,
            },
            {
                "platform": "NOAA-16",
                "start_time": np.datetime64("2009-07-01 02:15"),
                "end_time": np.datetime64("2009-07-01 02:30"),
                "along_track": 12000,
                "filename": "file7",
                "midnight_scanline": 1234,
                "overlap_free_start": np.nan,
                "overlap_free_start_exp": np.nan,
                "overlap_free_end": np.nan,
                "overlap_free_end_exp": np.nan,
                "global_quality_flag": QualityFlags.OK,
                "global_quality_flag_exp": QualityFlags.DUPLICATE,
            },
            {
                "platform": "NOAA-16",
                "start_time": np.datetime64("2009-07-01 03:00"),
                "end_time": np.datetime64("2009-07-01 03:01"),
                "along_track": 49,
                "filename": "file8",
                "midnight_scanline": 1234,
                "overlap_free_start": np.nan,
                "overlap_free_start_exp": np.nan,
                "overlap_free_end": np.nan,
                "overlap_free_end_exp": np.nan,
                "global_quality_flag": QualityFlags.OK,
                "global_quality_flag_exp": QualityFlags.TOO_SHORT,
            },
            {
                "platform": "NOAA-16",
                "start_time": np.datetime64("2009-07-01 02:50"),
                "end_time": np.datetime64("2009-07-01 04:00"),
                "along_track": 12000,
                "filename": "file9",
                "midnight_scanline": 1234,
                "overlap_free_start": np.nan,
                "overlap_free_start_exp": 2,
                "overlap_free_end": np.nan,
                "overlap_free_end_exp": 11999,
                "global_quality_flag": QualityFlags.OK,
                "global_quality_flag_exp": QualityFlags.OK,
            },
            {
                "platform": "NOAA-16",
                "start_time": np.datetime64("2009-07-01 03:50"),
                "end_time": np.datetime64("2008-07-01 05:00"),
                "along_track": 12000,
                "filename": "file10",
                "midnight_scanline": 1234,
                "overlap_free_start": np.nan,
                "overlap_free_start_exp": np.nan,
                "overlap_free_end": np.nan,
                "overlap_free_end_exp": np.nan,
                "global_quality_flag": QualityFlags.OK,
                "global_quality_flag_exp": QualityFlags.INVALID_TIMESTAMP,
            },
            {
                "platform": "NOAA-16",
                "start_time": np.datetime64("2009-07-01 04:50"),
                "end_time": np.datetime64("2009-07-01 05:59"),
                "along_track": 12000,
                "filename": "file12",
                "midnight_scanline": 1234,
                "overlap_free_start": np.nan,
                "overlap_free_start_exp": np.nan,
                "overlap_free_end": np.nan,
                "overlap_free_end_exp": np.nan,
                "global_quality_flag": QualityFlags.OK,
                "global_quality_flag_exp": QualityFlags.REDUNDANT,
            },
            {
                "platform": "NOAA-16",
                "start_time": np.datetime64("2009-07-01 04:50"),
                "end_time": np.datetime64("2009-07-01 06:00"),
                "along_track": 12000,
                "filename": "file11",
                "midnight_scanline": 1234,
                "overlap_free_start": np.nan,
                "overlap_free_start_exp": 0,
                "overlap_free_end": np.nan,
                "overlap_free_end_exp": 11999,
                "global_quality_flag": QualityFlags.OK,
                "global_quality_flag_exp": QualityFlags.OK,
            },
        ]

        # Add exact copy with another platform
        if multi_platform:
            noaa17 = [rec.copy() for rec in mda]
            for rec in noaa17:
                rec["platform"] = "NOAA-17"
            mda = mda + noaa17

        df = pd.DataFrame(mda)
        if reverse:
            df = df.sort_values(by=["start_time", "end_time"], ascending=False)
        return df

    @mock.patch("pygac_fdr.metadata.xr.open_dataset")
    def test_calc_overlap(self, open_dataset):
        open_dataset.side_effect = open_dataset_patched

        # Get test data and set quality flags as they affect the overlap computation
        mda = self.get_mda(multi_platform=False)
        mda.loc[:, "global_quality_flag"] = mda["global_quality_flag_exp"]

        # Check overlap computation (closed end)
        enhancer = MetadataEnhancer()
        mda_overlap = enhancer._calc_overlap_single_platform(mda.copy())
        pd.testing.assert_series_equal(
            mda_overlap["overlap_free_start"],
            mda_overlap["overlap_free_start_exp"],
            check_names=False,
        )
        pd.testing.assert_series_equal(
            mda_overlap["overlap_free_end"],
            mda_overlap["overlap_free_end_exp"],
            check_names=False,
        )

        # Open end
        mda_overlap_open_end = enhancer._calc_overlap_single_platform(
            mda.copy(), open_end=True
        )
        np.testing.assert_equal(
            mda_overlap_open_end.iloc[-1]["overlap_free_end"], np.nan
        )

    @mock.patch("pygac_fdr.metadata.xr.open_dataset")
    def test_enhance_metadata(self, open_dataset):
        open_dataset.side_effect = open_dataset_patched
        mda = self.get_mda(multi_platform=True, reverse=True)
        enhancer = MetadataEnhancer()
        mda = enhancer.enhance_metadata(mda)

        pd.testing.assert_series_equal(
            mda["global_quality_flag"],
            mda["global_quality_flag_exp"],
            check_names=False,
        )
        pd.testing.assert_series_equal(
            mda["overlap_free_start"], mda["overlap_free_start_exp"], check_names=False
        )
        pd.testing.assert_series_equal(
            mda["overlap_free_end"], mda["overlap_free_end_exp"], check_names=False
        )


class TestReadMetadata:
    @pytest.fixture(params=[["METOP-A"], ["METOP-A", "NOAA-16"]])
    def platforms(self, request):
        return request.param

    @pytest.fixture
    def metadata(self, platforms):
        mda = []
        for ip, platform in enumerate(platforms):
            mda.append(
                {"platform_index": ip, "file_index": 0, "platform": platform, "foo": 1}
            )
            mda.append(
                {"platform_index": ip, "file_index": 1, "platform": platform, "foo": 2}
            )
        df = pd.DataFrame(mda)
        return df.set_index(["platform_index", "file_index"], drop=True)

    @pytest.fixture
    def database(self, metadata, tmp_path):
        dbfile = tmp_path / "test.sqlite3"
        save_metadata_to_database(metadata, dbfile, "replace")
        return dbfile

    def test_read_metadata_from_database(self, database, metadata):
        res = read_metadata_from_database(database)
        pd.testing.assert_frame_equal(res, metadata)
