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
from distutils.version import StrictVersion
import logging
import os
import satpy
from string import Formatter
import xarray as xr
import pygac
import pygac_fdr
from pygac_fdr.utils import LOGGER_NAME

LOG = logging.getLogger(LOGGER_NAME)

DATASET_NAMES = {
    '1': 'reflectance_channel_1',
    '2': 'reflectance_channel_2',
    '3': 'reflectance_channel_3',
    '3a': 'reflectance_channel_3a',
    '3b': 'brightness_temperature_channel_3b',
    '4': 'brightness_temperature_channel_4',
    '5': 'brightness_temperature_channel_5',
}
FILL_VALUE_INT16 = -32767
FILL_VALUE_INT32 = -2147483648
DEFAULT_ENCODING = {
    'acq_time': {'units': 'seconds since 1970-01-01 00:00:00',
                 'calendar': 'standard'},
    'reflectance_channel_1': {'dtype': 'int16',
                              'scale_factor': 0.01,
                              'add_offset': 0,
                              '_FillValue': FILL_VALUE_INT16,
                              'zlib': True,
                              'complevel': 4},
    'reflectance_channel_2': {'dtype': 'int16',
                              'scale_factor': 0.01,
                              'add_offset': 0,
                              '_FillValue': FILL_VALUE_INT16,
                              'zlib': True,
                              'complevel': 4},
    'reflectance_channel_3': {'dtype': 'int16',
                              'scale_factor': 0.01,
                              'add_offset': 0,
                              '_FillValue': FILL_VALUE_INT16,
                              'zlib': True,
                              'complevel': 4},
    'reflectance_channel_3a': {'dtype': 'int16',
                               'scale_factor': 0.01,
                               'add_offset': 0,
                               '_FillValue': FILL_VALUE_INT16,
                               'zlib': True,
                               'complevel': 4},
    'brightness_temperature_channel_3b': {'dtype': 'int16',
                                          'scale_factor': 0.01,
                                          'add_offset': 273.15,
                                          '_FillValue': FILL_VALUE_INT16,
                                          'zlib': True,
                                          'complevel': 4},
    'brightness_temperature_channel_4': {'dtype': 'int16',
                                         'scale_factor': 0.01,
                                         'add_offset': 273.15,
                                         '_FillValue': FILL_VALUE_INT16,
                                         'zlib': True,
                                         'complevel': 4},
    'brightness_temperature_channel_5': {'dtype': 'int16',
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

METOP_PRE_LAUNCH_NUMBERS = {'a': 2, 'b': 1, 'c': 3}


def get_platform_short_name(pygac_name):
    """Get 3 character short name for the given platform."""
    if pygac_name.startswith('noaa'):
        nr = int(pygac_name[4:])
        return 'N{:02d}'.format(nr)
    elif pygac_name.startswith('metop'):
        nr = METOP_PRE_LAUNCH_NUMBERS[pygac_name[5:]]
        return 'M{:02d}'.format(nr)
    elif pygac_name == 'tirosn':
        # TODO: Correct?
        return 'N05'


def get_gcmd_platform_name(pygac_name):
    """Get platform name from NASA's Global Change Master Directory (GCMD).

    FUTURE: Use a library for this. Tried "pythesint" but installation failed.
    """
    if pygac_name.startswith('noaa'):
        nr = pygac_name[4:]
        gcmd_name = '{}-{}'.format(pygac_name[:4], nr).upper()
        gcmd_series = 'NOAA POES'
    elif pygac_name.startswith('metop'):
        letter = pygac_name[5:]
        gcmd_name = '{}-{}'.format(pygac_name[:5], letter).upper()
        gcmd_series = 'METOP'
    elif pygac_name == 'tirosn':
        gcmd_name = 'TIROS-N'
        gcmd_series = 'TIROS'
    else:
        raise ValueError('Invalid platform name: {}'.format(pygac_name))

    return '{} > {} > {}'.format('Earth Observation Satellites', gcmd_series, gcmd_name)


def get_gcmd_instrument_name(pygac_name):
    """Get instrument name from NASA's Global Change Master Directory (GCMD).

    FUTURE: Use a library for this. Tried "pythesint" but installation failed.
    """
    cat = 'Earth Remote Sensing Instruments > Passive Remote Sensing > ' \
          'Spectrometers/Radiometers > Imaging Spectrometers/Radiometers'
    return '{} > {}'.format(cat, pygac_name.upper())


class NetcdfWriter:
    """Write AVHRR GAC scenes to netCDF."""

    dataset_specific_attrs = ['units',
                              'wavelength',
                              'calibration',
                              'long_name',
                              'standard_name']
    shared_attrs = ['orbital_parameters']
    """Attributes shared between all datasets and to be included only once"""

    def_fname_fmt = 'avhrr_gac_fdr_{platform}_{start_time}_{end_time}.nc'
    time_fmt = '%Y%m%dT%H%M%SZ'
    def_engine = 'netcdf4'

    def __init__(self, global_attrs=None, encoding=None, engine=None, fname_fmt=None, debug=None):
        """
        Args:
            debug: If True, use constant creation time in output filenames.
        """
        self.global_attrs = global_attrs or {}
        self.engine = engine or self.def_engine
        self.fname_fmt = fname_fmt or self.def_fname_fmt
        self.debug = bool(debug)

        # User defined encoding takes precedence over default encoding
        self.encoding = DEFAULT_ENCODING.copy()
        self.encoding.update(encoding or {})

    def _get_temp_cov(self, scene):
        """Get temporal coverage of the dataset."""
        tstart = scene['4']['acq_time'][0]
        tend = scene['4']['acq_time'][-1]
        return tstart, tend

    def _get_integer_version(self, version):
        """Convert version string to integer.

        Examples:
             1.2.3 ->  123
            12.3.4 -> 1234

        Minor/patch versions > 9 are not supported.
        """
        numbers = StrictVersion(version).version
        if numbers[1] > 9 or numbers[2] > 9:
            raise ValueError('Invalid version number: {}. Minor/patch versions > 9 are not '
                             'supported'.format(version))
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
        tstart, tend = self._get_temp_cov(scene)
        platform = get_platform_short_name(scene['4'].attrs['platform_name'])
        version = self.global_attrs.get('version', '0.0.0')
        version_int = self._get_integer_version(version)
        fields = {'start_time': tstart.dt.strftime(self.time_fmt).data,
                  'end_time': tend.dt.strftime(self.time_fmt).data,
                  'platform': platform,
                  'version': version,
                  'version_int': version_int,
                  'creation_time': creation_time.strftime(self.time_fmt)}

        # Search for additional static fields in global attributes
        for _, field, _, _ in Formatter().parse(self.fname_fmt):
            if field and field not in fields:
                try:
                    fields[field] = self.global_attrs[field]
                except KeyError:
                    raise KeyError('Cannot find filename component {} in global attributes'.format(
                        field))

        return self.fname_fmt.format(**fields)

    def _get_global_attrs(self, scene):
        """Compile global attributes."""
        # Start with scene attributes
        global_attrs = scene.attrs.copy()

        # Transfer attributes shared by all channels (using channel 4 here, but could be any
        # channel)
        ch4 = scene['4']
        for attr in self.shared_attrs:
            global_attrs[attr] = ch4.attrs[attr]

        # Set some dynamic attributes
        tstart, tend = self._get_temp_cov(scene)
        resol = ch4.attrs['resolution']  # all channels have the same resolution
        global_attrs.update({
            'platform': get_gcmd_platform_name(ch4.attrs['platform_name']),
            'instrument': get_gcmd_instrument_name(ch4.attrs['sensor']),
            'date_created': datetime.now().isoformat(),
            'start_time': tstart.dt.strftime(self.time_fmt).data,
            'end_time': tend.dt.strftime(self.time_fmt).data,
            'sun_earth_distance_correction_factor': ch4.attrs['sun_earth_distance_correction_factor'],
            'version_pygac': pygac.__version__,
            'version_pygac_fdr': pygac_fdr.__version__,
            'version_satpy': satpy.__version__,
            'version_calib_coeffs': 'TODO',
            'geospatial_lon_min': scene['longitude'].min().values,
            'geospatial_lon_max': scene['longitude'].max().values,
            'geospatial_lon_units': 'degrees_east',
            'geospatial_lat_min': scene['latitude'].min().values,
            'geospatial_lat_max': scene['latitude'].max().values,
            'geospatial_lat_units': 'degrees_north',
            'geospatial_lon_resolution': '{} meters'.format(resol),
            'geospatial_lat_resolution': '{} meters'.format(resol)
        })
        global_attrs['time_coverage_start'] = global_attrs['start_time']
        global_attrs['time_coverage_end'] = global_attrs['end_time']
        global_attrs.pop('sensor')

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
        scn_keys = scn_keys.union(
            set([coord for key in scene.keys() for coord in scene[key].coords]))
        return dict([(key, self.encoding[key]) for key in enc_keys.intersection(scn_keys)])

    def _append_gac_header(self, filename, header):
        """Append raw GAC header to the given netCDF file."""
        data_vars = dict([(name, header[name]) for name in header.dtype.names])
        header = xr.Dataset(data_vars, attrs={'title': 'Raw GAC header'})
        header.to_netcdf(filename, mode='a', group='gac_header')

    def write(self, scene, output_dir=None):
        """Write an AVHRR GAC scene to netCDF.

        Args:
            scene (satpy.Scene): AVHRR GAC scene
            output_dir: Output directory. Filenames are composed dynamically based on the scene
                        contents.
        Returns:
            Names of files written.
        """
        output_dir = output_dir or '.'
        gac_header = scene['4'].attrs['gac_header'].copy()
        filename = os.path.join(output_dir, self._compose_filename(scene))
        global_attrs = self._get_global_attrs(scene)
        self._cleanup_attrs(scene)
        self._rename_datasets(scene)
        encoding = self._get_encoding(scene)
        LOG.info('Writing calibrated scene to {}'.format(filename))
        scene.save_datasets(writer='cf',
                            filename=filename,
                            header_attrs=global_attrs,
                            engine=self.engine,
                            flatten_attrs=True,
                            encoding=encoding,
                            pretty=True)
        self._append_gac_header(filename, gac_header)

        return filename
