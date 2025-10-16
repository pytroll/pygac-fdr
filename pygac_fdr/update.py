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

"""Update l1c metadata."""

import logging
import sqlite3
from datetime import datetime

import netCDF4
import numpy as np
import pandas as pd
import xarray as xr
from xarray.coding.times import encode_cf_datetime

from pygac_fdr.qflags import QualityFlags

LOG = logging.getLogger(__package__)


FILL_VALUE_INT = -9999
FILL_VALUE_FLOAT = -9999.0

ADDITIONAL_METADATA = [
    {
        "name": "overlap_free_start",
        "long_name": "First scanline (0-based) of the overlap-free part of this file. Scanlines "
        "before that also appear in the preceding file.",
        "dtype": np.int16,
        "fill_value": FILL_VALUE_INT,
    },
    {
        "name": "overlap_free_end",
        "long_name": "Last scanline (0-based) of the overlap-free part of this file. Scanlines "
        "hereafter also appear in the subsequent file.",
        "dtype": np.int16,
        "fill_value": FILL_VALUE_INT,
    },
    {
        "name": "midnight_line",
        "long_name": "Scanline (0-based) where UTC timestamp crosses the dateline",
        "dtype": np.int16,
        "fill_value": FILL_VALUE_INT,
    },
    {
        "name": "equator_crossing_longitude",
        "long_name": "Longitude where ascending node crosses the equator",
        "comment": "Can happen twice per file",
        "units": "degrees_east",
        "dtype": np.float64,
        "fill_value": FILL_VALUE_FLOAT,
    },
    {
        "name": "equator_crossing_time",
        "long_name": "UTC time when ascending node crosses the equator",
        "comment": "Can happen twice per file",
        "units": "seconds since 1970-01-01 00:00:00",
        "calendar": "standard",
        "dtype": np.float64,
        "fill_value": FILL_VALUE_FLOAT,
    },
    {
        "name": "global_quality_flag",
        "long_name": "Global quality flag",
        "comment": 'If this flag is everything else than "ok", it is recommended not '
        "to use the file.",
        "flag_values": np.array(
            [flag.value for flag in QualityFlags.__members__.values()], dtype=np.uint8
        ),
        "flag_meanings": " ".join(
            [name.lower() for name in QualityFlags.__members__.keys()]
        ),
        "dtype": np.uint8,
        "fill_value": None,
    },
]


def update_l1c_file_metadata(dbfile):
    """Update l1c file metadata with database contents."""
    mda = read_metadata_from_database(dbfile)
    updater = L1cFileMetadataUpdater()
    updater.update(mda)


def read_metadata_from_database(dbfile):
    """Read metadata from sqlite database."""
    with sqlite3.connect(dbfile) as con:
        mda = pd.read_sql("select * from metadata", con)
    mda = _restore_multi_index(mda)
    mda.fillna(value=np.nan, inplace=True)
    _cast_time_columns_to_datetime(mda)
    return mda


def _restore_multi_index(mda):
    return mda.set_index(["platform_index", "file_index"])


def _cast_time_columns_to_datetime(mda):
    for col in mda.columns:
        if "time" in col:
            mda[col] = mda[col].astype("datetime64[ns]")


class L1cFileMetadataUpdater:
    """Update metadata of l1c files."""

    def update(self, mda):
        """Update metadata of l1c files.

        Since xarray cannot modify files in-place, use netCDF4 directly. See
        https://github.com/pydata/xarray/issues/2029.
        """
        mda = self._to_xarray(mda)
        mda = self._stack(mda)
        for irow in range(mda.dims["row"]):
            row = mda.isel(row=irow)
            LOG.debug("Updating metadata in {}".format(row["filename"].item()))
            with netCDF4.Dataset(filename=row["filename"].item(), mode="r+") as nc:
                self._update_file(nc=nc, row=row)

    def _to_xarray(self, mda):
        """Convert pandas DataFrame to xarray Dataset."""
        mda = xr.Dataset(mda)
        mda = mda.rename({"dim_0": "row"})
        return mda

    def _stack(self, mda):
        """Stack certain columns to simplify the netCDF file."""
        # Stack equator crossing longitudes/times along a new dimension
        mda["equator_crossing_time"] = (
            ("row", "num_eq_cross"),
            np.stack(
                [
                    mda["equator_crossing_time_1"].values,
                    mda["equator_crossing_time_2"].values,
                ]
            ).transpose(),
        )
        mda["equator_crossing_longitude"] = (
            ("row", "num_eq_cross"),
            np.stack(
                [
                    mda["equator_crossing_longitude_1"].values,
                    mda["equator_crossing_longitude_2"].values,
                ]
            ).transpose(),
        )
        return mda

    def _create_nc_var(self, nc, var_name, fill_value, dtype, shape, dims):
        """Create netCDF variable and dimension."""
        # Create dimension if needed (only 1D at the moment)
        if dims:
            dim_name, size = dims[0], shape[0]
            if dim_name not in nc.dimensions:
                nc.createDimension(dim_name, size=size)

        # Create nc variable
        if var_name in nc.variables:
            nc_var = nc.variables[var_name]
        else:
            nc_var = nc.createVariable(
                var_name, datatype=dtype, fill_value=fill_value, dimensions=dims
            )
        return nc_var

    def _update_file(self, nc, row):
        """Update metadata of a single file."""
        for add_mda in ADDITIONAL_METADATA:
            add_mda = add_mda.copy()
            var_name = add_mda.pop("name")
            fill_value = add_mda.pop("fill_value")
            data = row[var_name]

            # Create nc variable
            nc_var = self._create_nc_var(
                nc=nc,
                var_name=var_name,
                fill_value=fill_value,
                dtype=add_mda.pop("dtype"),
                dims=data.dims,
                shape=data.shape,
            )

            # Write data to nc variable. Since netCDF4 cannot handle NaN nor NaT, disable
            # auto-masking, and set null-data to fill value manually.
            nc_var.set_auto_mask(False)
            if np.issubdtype(data.dtype, np.datetime64):
                data, _, _ = encode_cf_datetime(
                    data, units=add_mda["units"], calendar=add_mda["calendar"]
                )
                data = np.nan_to_num(data, nan=fill_value)
            else:
                data = data.fillna(fill_value).values
            nc_var[:] = data

            # Set attributes of nc variable
            for key, val in add_mda.items():
                nc_var.setncattr(key, val)
