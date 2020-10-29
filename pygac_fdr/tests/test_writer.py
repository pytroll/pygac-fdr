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
