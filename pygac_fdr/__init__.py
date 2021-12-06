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

try:
    from pygac_fdr.version import version as __version__  # noqa
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "No module named pygac_fdr.version. This could mean "
        "you didn't install 'pygac_fdr' properly. Try reinstalling ('pip "
        "install').")
try:
    # If the wheels of netCDF4 (used by this module) and h5py (imported by pygac) are incompatible,
    # segfaults or runtime errors like "NetCDF: HDF error" might occur. Prevent this by importing
    # netCDF4 first.
    import netCDF4  # noqa: F401
except ImportError:
    pass
