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

import unittest

import numpy as np
import pytest
import satpy
import xarray as xr

from pygac_fdr.writer import DEFAULT_ENCODING, NetcdfAttributeProcessor, NetcdfWriter


class NetcdfWriterTest(unittest.TestCase):
    def test_get_integer_version(self):
        writer = NetcdfWriter()
        self.assertEqual(writer._get_integer_version("1.2"), 120)
        self.assertEqual(writer._get_integer_version("1.2.3"), 123)
        self.assertEqual(writer._get_integer_version("12.3.4"), 1234)
        self.assertRaises(ValueError, writer._get_integer_version, "1.10.1")

    def test_default_encoding(self):
        bt_range = np.arange(170, 330, 1, dtype="f8")
        refl_range = np.arange(0, 1.5, 0.1, dtype="f8")
        test_data = {
            "reflectance_channel_1": refl_range,
            "reflectance_channel_2": refl_range,
            "brightness_temperature_channel_3": bt_range,
            "reflectance_channel_3a": refl_range,
            "brightness_temperature_channel_3b": bt_range,
            "brightness_temperature_channel_4": bt_range,
            "brightness_temperature_channel_5": bt_range,
        }
        for ch, data in test_data.items():
            enc = DEFAULT_ENCODING[ch]
            data_enc = ((data - enc["add_offset"]) / enc["scale_factor"]).astype(
                enc["dtype"]
            )
            data_dec = data_enc * enc["scale_factor"] + enc["add_offset"]
            np.testing.assert_allclose(data_dec, data, rtol=0.1)


class TestNetcdfAttributeProcessor:
    @pytest.fixture
    def scene(self):
        scene = satpy.Scene()
        acq_time = np.array([0, 1], dtype="datetime64[s]")
        scene["4"] = xr.DataArray(
            [[1, 2], [3, 4]],
            dims=("y", "x"),
            coords={
                "acq_time": ("y", acq_time),
            },
            attrs={
                "platform_name": "noaa15",
                "resolution": 1234.0,
                "sensor": "avhrr-1",
                "sun_earth_distance_correction_factor": 0.9,
                "calib_coeffs_version": "patmos-x 2012",
                "orbital_parameters": {"foo": "bar"},
            },
        )
        scene["latitude"] = xr.DataArray(
            [[1, 2], [3, 4]],
            dims=("y", "x"),
            coords={
                "acq_time": ("y", acq_time),
            },
        )
        scene["longitude"] = xr.DataArray(
            [[5, 6], [7, 8]],
            dims=("y", "x"),
            coords={
                "acq_time": ("y", acq_time),
            },
        )
        scene["qual_flags"] = xr.DataArray(
            [[0, 0], [0, 0]],
            dims=("y", "x"),
            coords={
                "acq_time": ("y", acq_time),
            },
        )
        return scene

    @pytest.fixture
    def attr_proc(self, scene):
        global_attrs = {"author": "Turtles"}
        return NetcdfAttributeProcessor(scene, global_attrs)

    def test_get_global_attrs(self, attr_proc):
        expected = {
            "author": "Turtles",
            "end_time": "19700101T000001Z",
            "geospatial_lat_max": 4,
            "geospatial_lat_min": 1,
            "geospatial_lat_resolution": "1234.0 meters",
            "geospatial_lat_units": "degrees_north",
            "geospatial_lon_max": 8,
            "geospatial_lon_min": 5,
            "geospatial_lon_resolution": "1234.0 meters",
            "geospatial_lon_units": "degrees_east",
            "instrument": "Earth Remote Sensing Instruments > Passive Remote Sensing > "
            "Spectrometers/Radiometers > Imaging Spectrometers/Radiometers "
            "> AVHRR-1",
            "orbital_parameters": {"foo": "bar"},
            "platform": "Earth Observation Satellites > NOAA POES > NOAA-15",
            "start_time": "19700101T000000Z",
            "sun_earth_distance_correction_factor": 0.9,
            "time_coverage_start": "19981026T005400Z",
            "version_calib_coeffs": "patmos-x 2012",
        }
        attrs = attr_proc.get_global_attrs()
        self._drop_dynamic_attrs(attrs)
        np.testing.assert_equal(attrs, expected)

    def _drop_dynamic_attrs(self, attrs):
        for drop_attrs in [
            "date_created",
            "version_satpy",
            "version_pygac",
            "version_pygac_fdr",
        ]:
            attrs.pop(drop_attrs)

    def test_update_attrs(self, attr_proc, scene):
        attr_proc.update_attrs(scene)
        assert "comment" in scene["qual_flags"].attrs
        expected = xr.DataArray(
            [[1, 2], [3, 4]],
            dims=("y", "x"),
            coords={
                "acq_time": xr.DataArray(
                    np.array([0, 1], dtype="datetime64[s]"),
                    dims="y",
                    attrs={"standard_name": "time", "axis": "T"},
                ),
                "y": xr.DataArray(
                    [0, 1], dims="y", attrs={"long_name": "Line number", "axis": "Y"}
                ),
                "x": xr.DataArray(
                    [0, 1], dims="x", attrs={"long_name": "Pixel number", "axis": "X"}
                ),
            },
            attrs={
                "name": "4",
                "resolution": 1234.0,
            },
        )
        scene["4"].attrs.pop("_satpy_id", None)
        xr.testing.assert_identical(scene["4"], expected)
