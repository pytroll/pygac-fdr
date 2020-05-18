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

from datetime import datetime
import os
import satpy


BANDS = ['1', '2', '3', '3a', '3b', '4', '5']
AUX_DATA = ['latitude',
            'longitude',
            'qual_flags',
            'sensor_zenith_angle',
            'solar_zenith_angle',
            'solar_azimuth_angle',
            'sensor_azimuth_angle',
            'sun_sensor_azimuth_difference_angle']
DATASET_NAMES = {
    '1': 'channel_1',
    '2': 'channel_2',
    '3': 'channel_3',
    '3a': 'channel_3a',
    '3b': 'channel_3b',
    '4': 'channel_4',
    '5': 'channel_5',
}
DEFAULT_ENCODING = {
    'channel_4': {'dtype': 'int16',
                  'scale_factor': 0.01,
                  '_FillValue': -32767,
                  'zlib': True,
                  'complevel': 4,
                  'add_offset': 273.15}
}  # refers to renamed datasets


def read_gac(filename, reader_kwargs):
    """Read AVHRR GAC scene using satpy.

    Args:
        filename (str): AVHRR GAC level 1b file
        reader_kwargs (dict): Keyword arguments to be passed to the reader.
    """
    scene = satpy.Scene(filenames=[filename], reader='avhrr_l1b_gaclac',
                        reader_kwargs=reader_kwargs)
    scene.load(BANDS)
    scene.load(AUX_DATA)

    # Add additional metadata
    scene.attrs['l1b_filename'] = os.path.basename(filename)
    filename_info = scene.readers['avhrr_l1b_gaclac'].file_handlers['gac_lac_l1b'][0].filename_info
    for key, val in filename_info.items():
        scene.attrs['l1b_' + key] = val

    return scene


class NetcdfWriter:
    """Write AVHRR GAC scenes to netCDF."""

    dataset_specific_attrs = ['units',
                              'wavelength',
                              'resolution',
                              'calibration',
                              'long_name',
                              'standard_name']
    shared_attrs = ['sensor',
                    'platform_name',
                    'start_time',
                    'end_time',
                    'orbital_parameters',
                    'orbit_number']

    def __init__(self, global_attrs=None, encoding=None, engine='netcdf4'):
        self.global_attrs = global_attrs if global_attrs is not None else {}
        self.engine = engine

        # User defined encoding takes precedence over default encoding
        self.encoding = DEFAULT_ENCODING.copy()
        self.encoding.update(encoding if encoding is not None else {})

    def _compose_filename(self, scene):
        """Compose output filename."""
        time_fmt = '%Y%m%d_%H%M%S'
        tstart = scene['4']['acq_time'].min()
        tend = scene['4']['acq_time'].max()
        return 'avhrr_gac_fdr_{}_{}_{}.nc'.format(scene['4'].attrs['platform_name'],
                                                  tstart.dt.strftime(time_fmt).data,
                                                  tend.dt.strftime(time_fmt).data)

    def _get_global_attrs(self, scene):
        """Compile global attributes."""
        # Start with scene attributes
        global_attrs = scene.attrs.copy()

        # Transfer more attributes from channel 4 (could be any channel, these information are
        # shared by all channels)
        for attr in self.shared_attrs:
            global_attrs[attr] = scene['4'].attrs[attr]

        # Set some dynamic attributes
        global_attrs.update({'creation_time': datetime.now().isoformat()})

        # User defined static attributes take precedence over dynamic attributes.
        global_attrs.update(self.global_attrs)

        return global_attrs

    def _cleanup_attrs(self, scene):
        """Cleanup attributes repeated in each dataset of the scene."""
        keep_attrs = self.dataset_specific_attrs + ['name', 'area']
        for ds in scene.keys():
            scene[ds].attrs = dict([(attr, val) for attr, val in scene[ds].attrs.items()
                                   if attr in keep_attrs])

    def _rename_datasets(self, scene):
        """Rename datasets in the scene to more verbose names."""
        for old_name, new_name in DATASET_NAMES.items():
            try:
                scene[new_name] = scene[old_name]
            except KeyError:
                continue
            del scene[old_name]

    def write(self, scene, output_dir):
        """Write an AVHRR GAC scene to netCDF.

        Args:
            scene (satpy.Scene): AVHRR GAC scene
            output_dir: Output directory. Filenames are composed dynamically based on the scene
                        contents.
        Returns:
            Names of files written.
        """
        filename = os.path.join(output_dir, self._compose_filename(scene))
        global_attrs = self._get_global_attrs(scene)
        self._cleanup_attrs(scene)
        self._rename_datasets(scene)
        scene.save_datasets(writer='cf',
                            filename=filename,
                            header_attrs=global_attrs,
                            engine=self.engine,
                            flatten_attrs=True,
                            encoding=self.encoding,
                            pretty=True)

        return filename
