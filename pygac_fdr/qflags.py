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

"""Quality flags."""

from enum import IntEnum


class QualityFlags(IntEnum):
    OK = 0
    INVALID_TIMESTAMP = 1  # end time < start time or timestamp out of valid range
    TOO_SHORT = 2  # not enough scanlines or duration too short
    TOO_LONG = 3  # (end_time - start_time) unrealistically large
    DUPLICATE = 4  # identical record from different ground stations
    REDUNDANT = 5  # subset of another file
