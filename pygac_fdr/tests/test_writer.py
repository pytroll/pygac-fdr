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
from pygac_fdr.writer import NetcdfWriter


class NetcdfWriterTest(unittest.TestCase):
    def test_get_integer_version(self):
        writer = NetcdfWriter()
        self.assertEqual(writer._get_integer_version('1.2'), 120)
        self.assertEqual(writer._get_integer_version('1.2.3'), 123)
        self.assertEqual(writer._get_integer_version('12.3.4'), 1234)
        self.assertRaises(ValueError, writer._get_integer_version, '1.10.1')
