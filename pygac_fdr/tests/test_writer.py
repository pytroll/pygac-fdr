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

import datetime as dt
import os
import unittest

import numpy as np
import pytest
import satpy
import xarray as xr
from pyresample.geometry import SwathDefinition
from satpy.tests.utils import make_dataid

from pygac_fdr.writer import DEFAULT_ENCODING, NetcdfWriter


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


class TestNetcdfWriter:
    @pytest.fixture
    def scene_lonlats(self):
        lons = [[5, 6], [7, 8]]
        lats = [[1, 2], [3, 4]]
        return lons, lats

    @pytest.fixture(params=[True, False])
    def with_orbital_parameters(self, request):
        return request.param

    @pytest.fixture
    def scene_dataset_attrs(self, scene_lonlats, with_orbital_parameters):
        lons, lats = scene_lonlats
        attrs = {
            "platform_name": "noaa15",
            "sensor": "avhrr-1",
            "sun_earth_distance_correction_factor": 0.9,
            "calib_coeffs_version": "patmos-x 2012",
            "long_name": "my_long_name",
            "units": "my_units",
            "standard_name": "my_standard_name",
            "area": SwathDefinition(lons, lats),
            "gac_header": np.array([(1, 2)], dtype=[("foo", "f4"), ("bar", "i4")]),
            "start_time": dt.datetime(2000, 1, 1),
            "end_time": dt.datetime(2000, 1, 1),
        }
        if with_orbital_parameters:
            attrs["orbital_parameters"] = {"tle": "my_tle"}
        return attrs

    @pytest.fixture
    def scene(self, scene_dataset_attrs, scene_lonlats):
        acq_time = np.array([0, 1], dtype="datetime64[s]")
        lons, lats = scene_lonlats
        scene = satpy.Scene()
        scene.attrs = {"sensor": "avhrr-1"}
        ch4_id = make_dataid(
            name="4",
            resolution=1234.0,
            wavelength="10.8um",
            modifiers=(),
            calibration="brightness_temperature",
        )
        lon_id = make_dataid(name="longitude", resolution=1234.0, modifiers=())
        lat_id = make_dataid(name="latitude", resolution=1234.0, modifiers=())
        qual_flags_id = make_dataid(name="qual_flags", resolution=1234.0, modifiers=())
        scene[ch4_id] = xr.DataArray(
            [[1, 2], [3, 4]],
            dims=("y", "x"),
            coords={
                "acq_time": ("y", acq_time),
            },
            attrs=scene_dataset_attrs,
        )
        scene[lat_id] = xr.DataArray(
            lats,
            dims=("y", "x"),
            coords={
                "acq_time": ("y", acq_time),
            },
        )
        scene[lon_id] = xr.DataArray(
            lons,
            dims=("y", "x"),
            coords={
                "acq_time": ("y", acq_time),
            },
        )
        scene[qual_flags_id] = xr.DataArray(
            [[0, 1, 0], [0, 0, 1]],
            dims=("y", "num_flags"),
            coords={
                "acq_time": ("y", acq_time),
            },
        )
        return scene

    @pytest.fixture
    def attrs_exp(self, with_orbital_parameters):
        attrs = {
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
            "platform": "Earth Observation Satellites > NOAA POES > NOAA-15",
            "start_time": "19700101T000000Z",
            "sun_earth_distance_correction_factor": 0.9,
            "time_coverage_start": "19981026T005400Z",
            "version_calib_coeffs": "patmos-x 2012",
            "Conventions": "CF-1.8",
            "product_version": "1.2.3",
        }
        if with_orbital_parameters:
            attrs["orbital_parameters_tle"] = "my_tle"
        return attrs

    @pytest.fixture
    def expected(self, attrs_exp):
        acq_time = xr.DataArray(
            np.array([0, 1], dtype="datetime64[s]"),
            dims="y",
            attrs={"standard_name": "time", "axis": "T"},
        )
        latitude = xr.DataArray(
            np.array([[1, 2], [3, 4]]),
            dims=("y", "x"),
            attrs={
                "name": "latitude",
                "standard_name": "latitude",
                "units": "degrees_north",
            },
        )
        longitude = xr.DataArray(
            np.array([[5, 6], [7, 8]]),
            dims=("y", "x"),
            attrs={
                "name": "longitude",
                "standard_name": "longitude",
                "units": "degrees_east",
            },
        )
        y = xr.DataArray(
            np.array([0, 1]),
            dims="y",
            attrs={"long_name": "Line number", "axis": "Y"},
        )
        x = xr.DataArray(
            np.array([0, 1]),
            dims="x",
            attrs={"long_name": "Pixel number", "axis": "X"},
        )
        bt_ch4 = xr.DataArray(
            [[1, 2], [3, 4]],
            dims=("y", "x"),
            coords={
                "acq_time": acq_time,
                "latitude": latitude,
                "longitude": longitude,
                "y": y,
                "x": x,
            },
            attrs={
                "units": "my_units",
                "wavelength": "10.8um",
                "calibration": "brightness_temperature",
                "long_name": "my_long_name",
                "standard_name": "my_standard_name",
                "resolution": 1234.0,
                "modifiers": [],
            },
        )
        qual_flags = xr.DataArray(
            [[0, 1, 0], [0, 0, 1]],
            dims=("y", "num_flags"),
            coords={
                "acq_time": acq_time,
                "y": y,
            },
            attrs={
                "long_name": "qual_flags",
                "comment": "Seven binary quality flags are provided per "
                "scanline. See the num_flags coordinate for their "
                "meanings.",
            },
        )

        return xr.Dataset(
            {"brightness_temperature_channel_4": bt_ch4, "qual_flags": qual_flags},
            attrs=attrs_exp,
        )

    @pytest.fixture
    def writer(self):
        user_defined_attrs = {
            "Conventions": "CF-1.8",
            "author": "Turtles",
            "product_version": "1.2.3",
        }
        return NetcdfWriter(global_attrs=user_defined_attrs)

    @pytest.fixture
    def output_file(self, writer, scene):
        filename = writer.write(scene)
        yield filename
        os.unlink(filename)

    def test_write(self, output_file, expected):
        with xr.open_dataset(output_file) as written:
            self._drop_dynamic_attrs(written.attrs)
            xr.testing.assert_identical(written, expected)

    def _drop_dynamic_attrs(self, attrs):
        for drop_attrs in [
            "history",
            "date_created",
            "version_satpy",
            "version_pygac",
            "version_pygac_fdr",
        ]:
            attrs.pop(drop_attrs)
