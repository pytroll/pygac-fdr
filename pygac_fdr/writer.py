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

"""Write AVHRR GAC level 1c data to netCDF."""

from datetime import datetime
import os
import satpy
import pygac_fdr.version


__version__ = pygac_fdr.version.__version__


DATASET_NAMES = {
    '1': 'channel_1',
    '2': 'channel_2',
    '3': 'channel_3',
    '3a': 'channel_3a',
    '3b': 'channel_3b',
    '4': 'channel_4',
    '5': 'channel_5',
}
FILL_VALUE_INT16 = -32767
FILL_VALUE_INT32 = -2147483648
DEFAULT_ENCODING = {
    'channel_1': {'dtype': 'int16',
                  'scale_factor': 0.01,
                  'add_offset': 0,
                  '_FillValue': FILL_VALUE_INT16,
                  'zlib': True,
                  'complevel': 4},
    'channel_2': {'dtype': 'int16',
                  'scale_factor': 0.01,
                  'add_offset': 0,
                  '_FillValue': FILL_VALUE_INT16,
                  'zlib': True,
                  'complevel': 4},
    'channel_3': {'dtype': 'int16',
                  'scale_factor': 0.01,
                  'add_offset': 0,
                  '_FillValue': FILL_VALUE_INT16,
                  'zlib': True,
                  'complevel': 4},
    'channel_3a': {'dtype': 'int16',
                   'scale_factor': 0.01,
                   'add_offset': 0,
                   '_FillValue': FILL_VALUE_INT16,
                   'zlib': True,
                   'complevel': 4},
    'channel_3b': {'dtype': 'int16',
                   'scale_factor': 0.01,
                   'add_offset': 273.15,
                   '_FillValue': FILL_VALUE_INT16,
                   'zlib': True,
                   'complevel': 4},
    'channel_4': {'dtype': 'int16',
                  'scale_factor': 0.01,
                  'add_offset': 273.15,
                  '_FillValue': FILL_VALUE_INT16,
                  'zlib': True,
                  'complevel': 4},
    'channel_5': {'dtype': 'int16',
                  'scale_factor': 0.01,
                  'add_offset': 273.15,
                  '_FillValue': FILL_VALUE_INT16,
                  'zlib': True,
                  'complevel': 4},
    'latitude': {'dtype': 'int32',
                 'scale_factor': 0.001,
                 'add_offset': 0,
                 '_FillValue': FILL_VALUE_INT32,
                 'zlib': True,
                 'complevel': 4},
    'longitude': {'dtype': 'int32',
                  'scale_factor': 0.001,
                  'add_offset': 0,
                  '_FillValue': FILL_VALUE_INT32,
                  'zlib': True,
                  'complevel': 4},
    'sensor_azimuth_angle': {'dtype': 'int16',
                             'scale_factor': 0.01,
                             'add_offset': 180.0,
                             '_FillValue': FILL_VALUE_INT16,
                             'zlib': True,
                             'complevel': 4},
    'sensor_zenith_angle': {'dtype': 'int16',
                            'scale_factor': 0.01,
                            'add_offset': 0,
                            '_FillValue': FILL_VALUE_INT16,
                            'zlib': True,
                            'complevel': 4},
    'solar_azimuth_angle': {'dtype': 'int16',
                            'scale_factor': 0.01,
                            'add_offset': 180.0,
                            '_FillValue': FILL_VALUE_INT16,
                            'zlib': True,
                            'complevel': 4},
    'solar_zenith_angle': {'dtype': 'int16',
                           'scale_factor': 0.01,
                           'add_offset': 0,
                           '_FillValue': FILL_VALUE_INT16,
                           'zlib': True,
                           'complevel': 4},
    'sun_sensor_azimuth_difference_angle': {'dtype': 'int16',
                                            'scale_factor': 0.01,
                                            'add_offset': 0,
                                            '_FillValue': FILL_VALUE_INT16,
                                            'zlib': True,
                                            'complevel': 4},
    'qual_flags': {'dtype': 'int16',
                   '_FillValue': FILL_VALUE_INT16,
                   'zlib': True,
                   'complevel': 4}
}  # refers to renamed datasets


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

    def_fname_fmt = 'avhrr_gac_fdr_v%(version)s_%(platform)s_%(start_time)s_%(end_time)s.nc'

    def __init__(self, global_attrs=None, encoding=None, engine='netcdf4', fname_fmt=None):
        self.global_attrs = global_attrs if global_attrs is not None else {}
        self.engine = engine
        self.fname_fmt = fname_fmt if fname_fmt is not None else self.def_fname_fmt

        # User defined encoding takes precedence over default encoding
        self.encoding = DEFAULT_ENCODING.copy()
        self.encoding.update(encoding if encoding is not None else {})

    def _compose_filename(self, scene):
        """Compose output filename."""
        time_fmt = '%Y%m%dT%H%M%SZ'
        tstart = scene['4']['acq_time'][0]
        tend = scene['4']['acq_time'][-1]
        fields = {'start_time': tstart.dt.strftime(time_fmt).data,
                  'end_time': tend.dt.strftime(time_fmt).data,
                  'platform': scene['4'].attrs['platform_name'],
                  'version': __version__.replace('.', '-')}
        return self.fname_fmt % fields

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

    def _get_encoding(self, scene):
        """Get netCDF encoding for the datasets in the scene."""
        # Remove entries from the encoding dictionary if the corresponding dataset is not available.
        # The CF writer doesn't like that.
        enc_keys = set(self.encoding.keys())
        scn_keys = set([key.name for key in scene.keys()])
        return dict([(key, self.encoding[key]) for key in enc_keys.intersection(scn_keys)])

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
        encoding = self._get_encoding(scene)
        scene.save_datasets(writer='cf',
                            filename=filename,
                            header_attrs=global_attrs,
                            engine=self.engine,
                            flatten_attrs=True,
                            encoding=encoding,
                            pretty=True)

        return filename
