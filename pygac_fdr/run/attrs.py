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

"""NetCDF attribute customization."""

from datetime import datetime

import numpy as np
import pygac
import satpy

import pygac_fdr
from pygac_fdr.metadata import TIME_COVERAGE
from pygac_fdr.run.utils import TIME_FMT, _get_temp_cov


class GlobalAttributeComposer:
    """Compose global attributes."""

    def __init__(self, scene, user_defined_attrs=None):
        self.scene = scene
        self.user_defined_attrs = user_defined_attrs or {}

    def get_global_attrs(self):
        """Compose global attributes."""
        global_attrs = self._copy_scene_attrs()
        global_attrs.update(self._compute_global_attrs())
        # User defined attributes take precedence.
        global_attrs.update(self.user_defined_attrs)
        return global_attrs

    def _copy_scene_attrs(self):
        scene_attrs = self.scene.attrs.copy()
        self._drop_unused_scene_attrs(scene_attrs)
        return scene_attrs

    def _drop_unused_scene_attrs(self, attrs):
        attrs.pop("sensor", None)  # we already have "instrument"

    def _compute_global_attrs(self):
        ch_attrs = self._get_channel_attrs()
        start_time, end_time = _get_temp_cov(self.scene)
        time_cov_start, time_cov_end = TIME_COVERAGE[
            get_gcmd_platform_name(ch_attrs["platform_name"], with_category=False)
        ]
        resol = ch_attrs["resolution"]  # all channels have the same resolution
        global_attrs = {
            "platform": get_gcmd_platform_name(ch_attrs["platform_name"]),
            "instrument": get_gcmd_instrument_name(ch_attrs["sensor"]),
            "date_created": datetime.now().isoformat(),
            "start_time": start_time.strftime(TIME_FMT).data,
            "end_time": end_time.strftime(TIME_FMT).data,
            "sun_earth_distance_correction_factor": ch_attrs[
                "sun_earth_distance_correction_factor"
            ],
            "version_pygac": pygac.__version__,
            "version_pygac_fdr": pygac_fdr.__version__,
            "version_satpy": satpy.__version__,
            "version_calib_coeffs": ch_attrs["calib_coeffs_version"],
            "geospatial_lon_min": self.scene["longitude"].min().values,
            "geospatial_lon_max": self.scene["longitude"].max().values,
            "geospatial_lon_units": "degrees_east",
            "geospatial_lat_min": self.scene["latitude"].min().values,
            "geospatial_lat_max": self.scene["latitude"].max().values,
            "geospatial_lat_units": "degrees_north",
            "geospatial_lon_resolution": "{} meters".format(resol),
            "geospatial_lat_resolution": "{} meters".format(resol),
            "time_coverage_start": time_cov_start.strftime(TIME_FMT),
            "orbital_parameters": ch_attrs.get("orbital_parameters", {}),
        }
        if time_cov_end:  # Otherwise still operational
            global_attrs["time_coverage_end"] = time_cov_end.strftime(TIME_FMT)
        return global_attrs

    def _get_channel_attrs(self):
        """Get channel attributes.

        Using channel 4 here, because that is available for all sensor
        generations.
        """
        return self.scene["4"].attrs


def get_gcmd_platform_name(pygac_name, with_category=True):
    """Get platform name from NASA's Global Change Master Directory (GCMD).

    FUTURE: Use a library for this. Tried "pythesint" but installation failed.
    """
    if pygac_name.startswith("noaa"):
        nr = pygac_name[4:]
        gcmd_name = "{}-{}".format(pygac_name[:4], nr).upper()
        gcmd_series = "NOAA POES"
    elif pygac_name.startswith("metop"):
        letter = pygac_name[5:]
        gcmd_name = "{}-{}".format(pygac_name[:5], letter).upper()
        gcmd_series = "METOP"
    elif pygac_name == "tirosn":
        gcmd_name = "TIROS-N"
        gcmd_series = "TIROS"
    else:
        raise ValueError("Invalid platform name: {}".format(pygac_name))

    if with_category:
        gcmd_name = "{} > {} > {}".format(
            "Earth Observation Satellites", gcmd_series, gcmd_name
        )

    return gcmd_name


def get_gcmd_instrument_name(pygac_name):
    """Get instrument name from NASA's Global Change Master Directory (GCMD).

    FUTURE: Use a library for this. Tried "pythesint" but installation failed.
    """
    cat = (
        "Earth Remote Sensing Instruments > Passive Remote Sensing > "
        "Spectrometers/Radiometers > Imaging Spectrometers/Radiometers"
    )
    return "{} > {}".format(cat, pygac_name.upper())


class AttributeProcessor:
    """Update dataset attributes."""

    dataset_specific_attrs = [
        "units",
        "wavelength",
        "calibration",
        "long_name",
        "standard_name",
        "name",
        "area",
        "_satpy_id",
    ]

    def update_attrs(self, scene):
        """Update dataset attributes."""
        self._cleanup_attrs_in_scene(scene)
        self._set_custom_attrs(scene)

    def _cleanup_attrs_in_scene(self, scene):
        """Cleanup attributes repeated in each dataset of the scene."""
        for ds in scene.keys():
            self._cleanup_attrs_in_dataset(scene[ds])

    def _cleanup_attrs_in_dataset(self, dataset):
        for drop_attr in self._get_attrs_to_be_dropped(dataset):
            dataset.attrs.pop(drop_attr)

    def _get_attrs_to_be_dropped(self, dataset):
        keep_attrs = self.dataset_specific_attrs.copy()
        if _has_xy_dims(dataset):
            keep_attrs.append("resolution")
        return set(dataset.attrs.keys()).difference(set(keep_attrs))

    def _set_custom_attrs(self, scene):
        """Set custom dataset attributes."""
        scene["qual_flags"].attrs["comment"] = (
            "Seven binary quality flags are provided per "
            "scanline. See the num_flags coordinate for their "
            "meanings."
        )


class CoordinateProcessor:
    """Update dataset coordinates."""

    latlon_attrs = ["units", "standard_name"]

    def update_coordinates(self, scene):
        """Update dataset coordinates.

        For each dataset with dimensions (y, x), add coordinates (y, x, lat,
        lon). This enables xr.to_netcdf() to set the proper coordinate
        attributes.
        """
        for ds_name in scene.keys():
            self._update_acq_time_coords(scene[ds_name])
            if _has_xy_dims(scene[ds_name]):
                self._add_latlon_coords(scene, ds_name)
                self._add_xy_coords(scene, ds_name)

    def _update_acq_time_coords(self, dataset):
        dataset["acq_time"].attrs.update({"standard_name": "time", "axis": "T"})

    def _add_latlon_coords(self, scene, ds_name):
        for coord_name in ("latitude", "longitude"):
            self._add_single_latlon_coord(scene, ds_name, coord_name)

    def _add_single_latlon_coord(self, scene, ds_name, coord_name):
        scene[ds_name].coords[coord_name] = (
            ("y", "x"),
            scene[coord_name].data,
        )
        self._add_latlon_coord_attrs(scene, ds_name, coord_name)

    def _add_latlon_coord_attrs(self, scene, ds_name, coord_name):
        scene[ds_name].coords[coord_name].attrs = dict(
            (key, val)
            for key, val in scene[coord_name].attrs.items()
            if key in self.latlon_attrs
        )

    def _add_xy_coords(self, scene, ds_name):
        scene[ds_name] = scene[ds_name].assign_coords(
            {
                "y": np.arange(scene[ds_name].shape[0]),
                "x": np.arange(scene[ds_name].shape[1]),
            }
        )
        self._update_xy_coord_attrs(scene, ds_name)

    def _update_xy_coord_attrs(self, scene, ds_name):
        scene[ds_name].coords["x"].attrs.update(
            {"axis": "X", "long_name": "Pixel number"}
        )
        scene[ds_name].coords["y"].attrs.update(
            {"axis": "Y", "long_name": "Line number"}
        )


def _has_xy_dims(dataset):
    return dataset.dims == ("y", "x")
