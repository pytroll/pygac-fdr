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


def read_gac(filename, reader_kwargs):
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


class PygacFdrNetcdfWriter:
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
    dataset_names = {
        '1': 'channel_1',
        '2': 'channel_2',
        '3': 'channel_3',
        '3a': 'channel_3a',
        '3b': 'channel_3b',
        '4': 'channel_4',
        '5': 'channel_5',
    }
    default_encoding = {
        'channel_4': {'dtype': 'int16',
                      'scale_factor': 0.01,
                      '_FillValue': -32767,
                      'zlib': True,
                      'complevel': 4,
                      'add_offset': 273.15}
    }  # refers to renamed datasets

    def __init__(self, global_attrs=None, encoding=None, engine='netcdf4'):
        self.global_attrs = global_attrs if global_attrs is not None else {}
        self.engine = engine

        # User defined encoding takes precedence over default encoding
        self.encoding = self.default_encoding.copy()
        self.encoding.update(encoding if encoding is not None else {})

    def _compose_filename(self, scene):
        time_fmt = '%Y%m%d_%H%M%S'
        tstart = scene['4']['acq_time'].min()
        tend = scene['4']['acq_time'].max()
        return 'avhrr_gac_fdr_{}_{}_{}.nc'.format(scene['4'].attrs['platform_name'],
                                                  tstart.dt.strftime(time_fmt).data,
                                                  tend.dt.strftime(time_fmt).data)

    def _get_global_attrs(self, scene):
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
        keep_attrs = self.dataset_specific_attrs + ['name', 'area']
        for ds in scene.keys():
            scene[ds].attrs = dict([(attr, val) for attr, val in scene[ds].attrs.items()
                                   if attr in keep_attrs])

    def _rename_datasets(self, scene):
        for ds_id in scene.keys():
            new_name = self.dataset_names.get(ds_id.name, ds_id.name)
            if new_name != ds_id.name:
                scene[new_name] = scene[ds_id]
                scene[new_name].attrs['name'] = new_name
                del scene[ds_id]

    def write(self, scene):
        filename = self._compose_filename(scene)
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
