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

"""NetCDF writer."""

import logging
import os
import warnings
from datetime import datetime
from distutils.version import StrictVersion
from string import Formatter

import netCDF4
import numpy as np
import satpy
import xarray as xr

from pygac_fdr.run.attrs import (
    AttributeProcessor,
    CoordinateProcessor,
    GlobalAttributeComposer,
)
from pygac_fdr.run.utils import TIME_FMT, _get_temp_cov
from pygac_fdr.utils import LOGGER_NAME

LOG = logging.getLogger(LOGGER_NAME)

DATASET_NAMES = {
    "1": "reflectance_channel_1",
    "2": "reflectance_channel_2",
    "3": "brightness_temperature_channel_3",
    "3a": "reflectance_channel_3a",
    "3b": "brightness_temperature_channel_3b",
    "4": "brightness_temperature_channel_4",
    "5": "brightness_temperature_channel_5",
}

FILL_VALUE_INT16 = -32767
FILL_VALUE_INT32 = -2147483648
DEFAULT_ENCODING = {
    "y": {"dtype": "int16"},
    "x": {"dtype": "int16"},
    "acq_time": {
        "units": "seconds since 1970-01-01 00:00:00",
        "calendar": "standard",
        "_FillValue": None,
    },
    "reflectance_channel_1": {
        "dtype": "int16",
        "scale_factor": 0.01,
        "_FillValue": FILL_VALUE_INT16,
        "zlib": True,
        "complevel": 4,
    },
    "reflectance_channel_2": {
        "dtype": "int16",
        "scale_factor": 0.01,
        "_FillValue": FILL_VALUE_INT16,
        "zlib": True,
        "complevel": 4,
    },
    "brightness_temperature_channel_3": {
        "dtype": "int16",
        "scale_factor": 0.01,
        "add_offset": 273.15,
        "_FillValue": FILL_VALUE_INT16,
        "zlib": True,
        "complevel": 4,
    },
    "reflectance_channel_3a": {
        "dtype": "int16",
        "scale_factor": 0.01,
        "_FillValue": FILL_VALUE_INT16,
        "zlib": True,
        "complevel": 4,
    },
    "brightness_temperature_channel_3b": {
        "dtype": "int16",
        "scale_factor": 0.01,
        "add_offset": 273.15,
        "_FillValue": FILL_VALUE_INT16,
        "zlib": True,
        "complevel": 4,
    },
    "brightness_temperature_channel_4": {
        "dtype": "int16",
        "scale_factor": 0.01,
        "add_offset": 273.15,
        "_FillValue": FILL_VALUE_INT16,
        "zlib": True,
        "complevel": 4,
    },
    "brightness_temperature_channel_5": {
        "dtype": "int16",
        "scale_factor": 0.01,
        "add_offset": 273.15,
        "_FillValue": FILL_VALUE_INT16,
        "zlib": True,
        "complevel": 4,
    },
    "latitude": {
        "dtype": "int32",
        "scale_factor": 0.001,
        "_FillValue": FILL_VALUE_INT32,
        "zlib": True,
        "complevel": 4,
    },
    "longitude": {
        "dtype": "int32",
        "scale_factor": 0.001,
        "_FillValue": FILL_VALUE_INT32,
        "zlib": True,
        "complevel": 4,
    },
    "sensor_azimuth_angle": {
        "dtype": "int16",
        "scale_factor": 0.01,
        "_FillValue": FILL_VALUE_INT16,
        "zlib": True,
        "complevel": 4,
    },
    "sensor_zenith_angle": {
        "dtype": "int16",
        "scale_factor": 0.01,
        "_FillValue": FILL_VALUE_INT16,
        "zlib": True,
        "complevel": 4,
    },
    "solar_azimuth_angle": {
        "dtype": "int16",
        "scale_factor": 0.01,
        "_FillValue": FILL_VALUE_INT16,
        "zlib": True,
        "complevel": 4,
    },
    "solar_zenith_angle": {
        "dtype": "int16",
        "scale_factor": 0.01,
        "_FillValue": FILL_VALUE_INT16,
        "zlib": True,
        "complevel": 4,
    },
    "sun_sensor_azimuth_difference_angle": {
        "dtype": "int16",
        "scale_factor": 0.01,
        "_FillValue": FILL_VALUE_INT16,
        "zlib": True,
        "complevel": 4,
    },
    "qual_flags": {
        "dtype": "int16",
        "_FillValue": FILL_VALUE_INT16,
        "zlib": True,
        "complevel": 4,
    },
    "num_flags": {"dtype": "<S50"},
}  # refers to renamed datasets
METOP_PRE_LAUNCH_NUMBERS = {"a": 2, "b": 1, "c": 3}


def get_platform_short_name(pygac_name):
    """Get 3 character short name for the given platform."""
    if pygac_name.startswith("noaa"):
        nr = int(pygac_name[4:])
        return "N{:02d}".format(nr)
    elif pygac_name.startswith("metop"):
        nr = METOP_PRE_LAUNCH_NUMBERS[pygac_name[5:]]
        return "M{:02d}".format(nr)
    elif pygac_name == "tirosn":
        return "TSN"


class NetcdfWriter:
    """Write AVHRR GAC scenes to netCDF."""

    dataset_specific_attrs = [
        "units",
        "wavelength",
        "calibration",
        "long_name",
        "standard_name",
    ]
    shared_attrs = ["orbital_parameters"]
    """Attributes shared between all datasets and to be included only once"""

    def_fname_fmt = "avhrr_gac_fdr_{platform}_{start_time}_{end_time}.nc"
    def_engine = "netcdf4"

    def __init__(
        self,
        output_dir=None,
        global_attrs=None,
        gac_header_attrs=None,
        encoding=None,
        engine=None,
        fname_fmt=None,
        debug=None,
    ):
        """
        Args:
            output_dir (str): Specifies the output directory. Default: Current
                directory.
            global_attrs (dict): User-defined global attributes to be included.
            gac_header_attrs (dict): Attributes describing the raw GAC header.
            encoding (dict): Specifies how to encode the datasets. See
                https://xarray.pydata.org/en/stable/user-guide/io.html?highlight=encoding#writing-encoded-data
            engine (str): NetCDF engine to be used. Default: netcdf4
            fname_fmt (str): Specifies the filename format.
            debug (bool): If True, use constant creation time in output filenames.
        """
        self.output_dir = output_dir or "."
        self.global_attrs = global_attrs or {}
        self.gac_header_attrs = gac_header_attrs or {}
        self.engine = engine or self.def_engine
        self.fname_fmt = fname_fmt or self.def_fname_fmt
        self.debug = bool(debug)

        # User defined encoding takes precedence over default encoding
        self.encoding = DEFAULT_ENCODING.copy()
        self.encoding.update(encoding or {})

    def _get_integer_version(self, version):
        """Convert version string to integer.

        Examples:
             1.2.3 ->  123
            12.3.4 -> 1234

        Minor/patch versions > 9 are not supported.
        """
        numbers = StrictVersion(version).version
        if numbers[1] > 9 or numbers[2] > 9:
            raise ValueError(
                "Invalid version number: {}. Minor/patch versions > 9 are not "
                "supported".format(version)
            )
        return sum(10**i * v for i, v in enumerate(reversed(numbers)))

    def _compose_filename(self, scene):
        """Compose output filename."""
        # Dynamic fields
        if self.debug:
            # In debug mode, use a constant creation time to prevent a different filename in each
            # run
            creation_time = datetime(2020, 1, 1)
        else:
            creation_time = datetime.now()
        start_time, end_time = _get_temp_cov(scene)
        platform = get_platform_short_name(scene["4"].attrs["platform_name"])
        try:
            version = self.global_attrs["product_version"]
        except KeyError:
            version = "0.0.0"
            msg = "No product_version set in global attributes. Falling back to 0.0.0"
            LOG.warning(msg)
            warnings.warn(msg)
        version_int = self._get_integer_version(version)
        fields = {
            "start_time": start_time.strftime(TIME_FMT).data,
            "end_time": end_time.strftime(TIME_FMT).data,
            "platform": platform,
            "version": version,
            "version_int": version_int,
            "creation_time": creation_time.strftime(TIME_FMT),
        }

        # Search for additional static fields in global attributes
        for _, field, _, _ in Formatter().parse(self.fname_fmt):
            if field and field not in fields:
                try:
                    fields[field] = self.global_attrs[field]
                except KeyError:
                    raise KeyError(
                        "Cannot find filename component {} in global attributes".format(
                            field
                        )
                    )

        return self.fname_fmt.format(**fields)

    def _rename_datasets(self, scene):
        """Rename datasets in the scene to more verbose names."""
        for old_name, new_name in DATASET_NAMES.items():
            try:
                scene[new_name] = scene[old_name]
            except KeyError:
                continue
            del scene[old_name]

    def _get_encoding(self, scene):
        """Get netCDF encoding for the datasets in the scene."""
        enc = SceneEncoder(self.encoding)
        return enc.get_encoding(scene)

    def _fix_global_attrs(self, filename, global_attrs):
        LOG.info("Fixing global attributes")
        with netCDF4.Dataset(filename, mode="a") as nc:
            # Satpy's CF writer overrides Conventions attribute
            nc.Conventions = global_attrs["Conventions"]

            # Satpy's CF writer assumes x/y to be projection coordinates
            for var_name in ("x", "y"):
                for drop_attr in ["standard_name", "units"]:
                    nc.variables[var_name].delncattr(drop_attr)

    def _append_gac_header(self, filename, header):
        """Append raw GAC header to the given netCDF file."""
        LOG.info("Appending GAC header")
        data_vars = dict([(name, header[name]) for name in header.dtype.names])
        header = xr.Dataset(data_vars, attrs=self.gac_header_attrs)
        header.to_netcdf(filename, mode="a", group="gac_header")

    def write(self, scene):
        """Write an AVHRR GAC scene to netCDF.

        Args:
            scene (satpy.Scene): AVHRR GAC scene
        Returns:
            Names of files written.
        """
        filename = os.path.join(self.output_dir, self._compose_filename(scene))
        gac_header = self._get_gac_header(scene)
        global_attrs = self._get_global_attrs(scene)
        self._preproc_scene(scene)
        self._save_datasets(scene, filename, global_attrs)
        self._postproc_file(filename, gac_header, global_attrs)
        return filename

    def _get_gac_header(self, scene):
        return scene["4"].attrs["gac_header"].copy()

    def _get_global_attrs(self, scene):
        ac = GlobalAttributeComposer(scene, self.global_attrs)
        return ac.get_global_attrs()

    def _preproc_scene(self, scene):
        CoordinateProcessor().update_coordinates(scene)
        AttributeProcessor().update_attrs(scene)
        self._rename_datasets(scene)

    def _save_datasets(self, scene, filename, global_attrs):
        encoding = self._get_encoding(scene)
        LOG.info("Writing calibrated scene to {}".format(filename))
        scene.save_datasets(
            writer="cf",
            filename=filename,
            header_attrs=global_attrs,
            engine=self.engine,
            flatten_attrs=True,
            encoding=encoding,
            pretty=True,
        )

    def _postproc_file(self, filename, gac_header, global_attrs):
        self._append_gac_header(filename, gac_header)
        self._fix_global_attrs(filename, global_attrs)


class SceneEncoder:
    def __init__(self, encoding):
        self.encoding = encoding

    def get_encoding(self, scene):
        enc = self._get_encoding_for_available_datasets(scene)
        self._fix_dtypes(enc)
        return enc

    def _get_encoding_for_available_datasets(self, scene):
        common_keys = self._get_keys_in_both_scene_and_encoding(scene)
        return dict([(key, self.encoding[key]) for key in common_keys])

    def _get_keys_in_both_scene_and_encoding(self, scene):
        scn_keys = self._get_scene_keys(scene)
        enc_keys = set(self.encoding.keys())
        return enc_keys.intersection(scn_keys)

    def _get_scene_keys(self, scene):
        dataset_keys = set([key["name"] for key in scene.keys()])
        coords_keys = set(
            [coord for key in scene.keys() for coord in scene[key].coords]
        )
        return dataset_keys.union(coords_keys)

    def _fix_dtypes(self, encoding):
        for enc in encoding.values():
            if "scale_factor" in enc:
                enc["scale_factor"] = np.float64(enc["scale_factor"])
            if "add_offset" in enc:
                enc["add_offset"] = np.float64(enc["add_offset"])
