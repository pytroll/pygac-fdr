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

"""End-to-end tests for pygac-fdr.

Download test data set, run the entire chain of processing steps (read GAC files, write data
to netCDF, enhance metadata) and compare results against reference data. They should be identical.

Usage:

$ python test_end2end.py

The test behaviour can be controlled using the configuration file test_end2end.yaml .

Do not use pytest as this will eat up all your memory, because pytest caches all generated figures
until the end of the test suite for some reason.
"""

from cfchecker.cfchecks import CFChecker
from dateutil.parser import isoparse
import glob
import gzip
import logging
import numpy as np
import os
import shutil
import subprocess
import unittest
import xarray as xr
from xarray.core.utils import dict_equiv
from xarray.core.formatting import diff_attrs_repr
import yaml

from pygac_fdr.metadata import QualityFlags
from pygac_fdr.utils import logging_on, LOGGER_NAME


LOG = logging.getLogger(LOGGER_NAME)


def call_subproc(cmd, cwd='.'):
    LOG.info('Calling subprocess {}'.format(cmd))
    ret = subprocess.call(cmd, cwd=cwd)
    if ret:
        raise RuntimeError('Subprocess {} failed with return code {}'.format(cmd, ret))


class EndToEndTestBase(unittest.TestCase):
    tag = None
    with_metadata = False

    def _assert_time_attrs_close(self, attrs_a, attrs_b):
        time_attrs = ['start_time', 'end_time']
        for attr in time_attrs:
            with self.subTest(attribute=attr):
                time_a = isoparse(attrs_a.pop(attr))
                time_b = isoparse(attrs_b.pop(attr))
                self.assertLess(abs((time_b - time_a).total_seconds()), 2)

    def _assert_numerical_attrs_close(self, attrs_a, attrs_b):
        numerical_attrs = [
            'geospatial_lon_min',
            'geospatial_lon_max',
            'geospatial_lat_min',
            'geospatial_lat_max',
        ]
        for attr in numerical_attrs:
            with self.subTest(attribute=attr):
                val_a = attrs_a.pop(attr)
                val_b = attrs_b.pop(attr)
                np.testing.assert_allclose(val_a, val_b)

    def assert_attrs_close(self, a, b):
        attrs_a = a.attrs.copy()
        attrs_b = b.attrs.copy()
        self._assert_time_attrs_close(attrs_a, attrs_b)
        self._assert_numerical_attrs_close(attrs_a, attrs_b)
        assert dict_equiv(attrs_a, attrs_b), diff_attrs_repr(attrs_a, attrs_b, 'identical')

    @classmethod
    def setUpClass(cls):
        logging_on(logging.DEBUG, for_all=True)

        # Read config file
        with open(os.path.join(os.path.dirname(__file__), 'test_end2end.yaml')) as fh:
            config = yaml.safe_load(fh)
        cls.test_data_dir = config['test_data_dir']
        cls.cleanup = config['cleanup']
        cls.resume = config['resume']
        cls.fast = config['fast']
        cls.rtol = float(config['rtol'])
        cls.atol = float(config['atol'])
        cls.plot = config['plot']

        # Set class attributes
        cls.cfg_file = os.path.join(os.path.dirname(__file__), '../../etc/pygac-fdr.yaml')
        cls.tle_dir = os.path.join(cls.test_data_dir, 'tle')
        cls.input_dir = os.path.join(cls.test_data_dir, 'input', cls.tag)
        cls.output_dir = os.path.join(cls.test_data_dir, 'output', cls.tag)
        cls.output_dir_ref = os.path.join(cls.test_data_dir, 'output_ref', cls.tag)

        # Discover input files and reference output files
        cls.gac_files_gz, cls.nc_files_ref = cls._discover_files()

        # Process input files
        dbfile = os.path.join(cls.output_dir, 'test.sqlite3')
        cls.nc_files = cls._run(cls.gac_files_gz,
                                dbfile=dbfile if cls.with_metadata else None)

    @classmethod
    def _discover_files(cls):
        gac_files_gz = sorted(
            glob.glob(os.path.join(cls.input_dir, '*.gz'))
        )
        nc_files_ref = sorted(
            glob.glob(os.path.join(cls.output_dir_ref, '*.nc'))
        )
        if cls.fast:
            gac_files_gz = [gac_files_gz[-1]]
            nc_files_ref = [nc_files_ref[-1]]
        return gac_files_gz, nc_files_ref

    @classmethod
    def _unzip_gac_files(cls, filenames_gz, output_dir):
        filenames = []
        for filename_gz in filenames_gz:
            basename_gz = os.path.basename(filename_gz)
            basename = os.path.splitext(basename_gz)[0]
            filename = os.path.join(output_dir, os.path.basename(basename))
            LOG.info('Decompressing {}'.format(filename_gz))
            with gzip.open(filename_gz, 'rb') as f_in, \
                 open(filename, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            filenames.append(filename)
        return filenames

    @classmethod
    def _run(cls, gac_files_gz, dbfile=None):
        """Read GAC files and write calibrated scenes to netCDF.

        If database file is given, also collect & update metadata.
        """
        if cls.resume:
            # Resume testing with existing output files
            LOG.info('Resuming test with existing output files')
            return sorted(
                glob.glob(os.path.join(cls.output_dir, '*.nc'))
            )

        # Prepare output directory
        if not os.path.isdir(cls.output_dir):
            os.makedirs(cls.output_dir)
        cls._cleanup_output_dir()

        # Unzip GAC files
        gac_files = cls._unzip_gac_files(gac_files_gz, cls.output_dir)

        # Read GAC files and write netCDF files
        run = ['pygac-fdr-run',
               '--cfg', cls.cfg_file,
               '--output-dir', cls.output_dir,
               '--tle-dir', cls.tle_dir,
               '--verbose', '--log-all'] + gac_files
        call_subproc(run)
        nc_files = sorted(glob.glob(cls.output_dir + '/*.nc'))

        if dbfile:
            # Collect & complement metadata
            collect = ['pygac-fdr-mda-collect', '--dbfile', dbfile, '--if-exists', 'replace',
                       '--verbose'] + nc_files
            call_subproc(collect)

            # Update metadata
            update = ['pygac-fdr-mda-update', '--dbfile', dbfile]
            call_subproc(update)

        return nc_files

    @classmethod
    def _cleanup_output_dir(cls):
        if cls.cleanup:
            for item in os.listdir(cls.output_dir):
                item = os.path.join(cls.output_dir, item)
                if os.path.isfile(item):
                    os.unlink(item)

    @classmethod
    def tearDownClass(cls):
        cls._cleanup_output_dir()

    def _tst_regression(self, nc_files, nc_files_ref):
        """Test entire netCDF contents against reference file."""
        dynamic_attrs = ['date_created', 'history', 'version_satpy', 'version_pygac',
                         'version_pygac_fdr']
        for nc_file, nc_file_ref in zip(nc_files, nc_files_ref):
            LOG.info('Performing regression test with {}'.format(nc_file))
            with self.subTest(nc_file=nc_file), \
                 xr.open_dataset(nc_file) as ds, \
                 xr.open_dataset(nc_file_ref) as ds_ref:

                # Remove dynamic attributes
                for attr in dynamic_attrs:
                    ds.attrs.pop(attr)
                    ds_ref.attrs.pop(attr)

                # If testing just one file, there is no overlap
                if self.fast:
                    ds = ds.drop_vars(['overlap_free_start', 'overlap_free_end'],
                                      errors='ignore')
                    ds_ref = ds_ref.drop_vars(['overlap_free_start', 'overlap_free_end'],
                                              errors='ignore')

                # Compare datasets
                self.assert_attrs_close(ds, ds_ref)
                try:
                    xr.testing.assert_allclose(ds, ds_ref, atol=self.atol, rtol=self.rtol)
                except AssertionError:
                    if self.plot:
                        self._plot_diffs(ds_ref=ds_ref, ds_tst=ds, file_tst=nc_file)
                    raise

    def _plot_diffs(self, ds_ref, ds_tst, file_tst):
        """Plot differences and save figure to output directory."""
        import matplotlib.pyplot as plt

        cmp_vars = {
            'reflectance_channel_1': {},
            'reflectance_channel_2': {},
            'brightness_temperature_channel_3': {},
            'reflectance_channel_3a': {},
            'brightness_temperature_channel_3b': {},
            'brightness_temperature_channel_4': {},
            'brightness_temperature_channel_5': {},
            'longitude': {'vmin': -180, 'vmax': 180},
            'latitude': {'vmin': -90, 'vmax': 90},
            'solar_zenith_angle': {},
            'solar_azimuth_angle': {},
            'sensor_zenith_angle': {},
            'sensor_azimuth_angle': {},
            'sun_sensor_azimuth_difference_angle': {},
        }

        for varname, prop in cmp_vars.items():
            try:
                diff = ds_tst[varname] - ds_ref[varname]
            except KeyError:
                continue
            vmin = min(ds_tst[varname].min(), ds_ref[varname].min())
            vmax = max(ds_tst[varname].max(), ds_ref[varname].max())
            min_diff = diff.min()
            max_diff = diff.max()
            abs_max_diff = max(abs(min_diff), abs(max_diff))

            fig, (ax_ref, ax_tst, ax_diff) = plt.subplots(nrows=3,
                                                          figsize=(20, 8.5),
                                                          sharex=True)
            ds_ref[varname].transpose().plot.imshow(
                ax=ax_ref, vmin=prop.get('vmin', vmin), vmax=prop.get('vmax', vmax)
            )
            ds_tst[varname].transpose().plot.imshow(
                ax=ax_tst, vmin=prop.get('vmin', vmin), vmax=prop.get('vmax', vmax)
            )
            diff.transpose().plot.imshow(
                ax=ax_diff, vmin=-0.8 * abs_max_diff, vmax=0.8 * abs_max_diff, cmap='RdBu_r'
            )

            ax_ref.set_title('Reference', fontsize=10)
            ax_tst.set_title('Test', fontsize=10)
            ax_diff.set_title('Test - Reference', fontsize=10)
            ax_ref.set_xlabel(None)
            ax_tst.set_xlabel(None)
            plt.suptitle(os.path.basename(file_tst))

            ofile = os.path.join(self.output_dir,
                                 '{}_{}.png'.format(os.path.basename(file_tst), varname))
            plt.savefig(ofile, bbox_inches='tight')
            plt.close('all')


class EndToEndTestNormal(EndToEndTestBase):
    """End-to-end test with normal data (no corruption).

    Also compare metadata against results from CLARA-A3 feedback loop 1.
    """

    tag = 'normal'
    with_metadata = True
    mda_exp = {
        'NSS.GHRR.NA.D81089.S0054.E0246.B0912021.GC': {
            'nc_file': 'AVHRR-GAC_FDR_1C_N06_19810330T005421Z_19810330T024632Z_R_O_20200101T000000Z_0100.nc',
            'midnight_line': np.nan,
            'overlap_free_end': 12995,
            'global_quality_flag': QualityFlags.OK
        },
        'NSS.GHRR.NA.D81089.S0242.E0427.B0912122.GC': {
            'nc_file': 'AVHRR-GAC_FDR_1C_N06_19810330T024239Z_19810330T042759Z_R_O_20200101T000000Z_0100.nc',
            'midnight_line': np.nan,
            'overlap_free_end': 12157,
            'global_quality_flag': QualityFlags.OK
        },
        'NSS.GHRR.NA.D81089.S0423.E0609.B0912223.GC': {
            'nc_file': 'AVHRR-GAC_FDR_1C_N06_19810330T042358Z_19810330T060903Z_R_O_20200101T000000Z_0100.nc',
            'midnight_line': np.nan,
            'overlap_free_end': 12100,
            # 12602 (=along_track-1) in CLARA, but that's because the next file was skipped
            # for some reason
            'global_quality_flag': QualityFlags.OK
        },
        'NSS.GHRR.NA.D81089.S0604.E0758.B0912324.WI': {
            'nc_file': 'AVHRR-GAC_FDR_1C_N06_19810330T060453Z_19810330T075842Z_R_O_20200101T000000Z_0100.nc',
            'midnight_line': np.nan,
            'overlap_free_end': 13020,  # Skipped in CLARA
            'global_quality_flag': QualityFlags.OK
        },
        'NSS.GHRR.NA.D81089.S0753.E0947.B0912425.WI': {
            'nc_file': 'AVHRR-GAC_FDR_1C_N06_19810330T075339Z_19810330T094742Z_R_O_20200101T000000Z_0100.nc',
            'midnight_line': np.nan,
            'overlap_free_end': 13053,
            # 13618 (=along_track-1) in CLARA, but that's because only the next file
            # (redundant in this case) is taken into account. It is not taken into
            # account that there might be overlap with the file after that.
            'global_quality_flag': QualityFlags.OK
        },
        'NSS.GHRR.NA.D81089.S0943.E1058.B0912525.WI': {
            'nc_file': 'AVHRR-GAC_FDR_1C_N06_19810330T094300Z_19810330T105846Z_R_O_20200101T000000Z_0100.nc',
            'midnight_line': np.nan,
            'overlap_free_end': np.nan,
            'global_quality_flag': QualityFlags.REDUNDANT
        },
        'NSS.GHRR.NA.D81089.S0943.E1136.B0912526.WI': {
            'nc_file': 'AVHRR-GAC_FDR_1C_N06_19810330T094300Z_19810330T113615Z_R_O_20200101T000000Z_0100.nc',
            'midnight_line': np.nan,
            'overlap_free_end': 12989,  # Skipped in CLARA
            'global_quality_flag': QualityFlags.OK
        },
        'NSS.GHRR.NA.D81089.S1102.E1136.B0912626.WI': {
            'nc_file': 'AVHRR-GAC_FDR_1C_N06_19810330T110243Z_19810330T113615Z_R_O_20200101T000000Z_0100.nc',
            'midnight_line': np.nan,
            'overlap_free_end': np.nan,
            'global_quality_flag': QualityFlags.REDUNDANT
        },
        'NSS.GHRR.NA.D81089.S1131.E1205.B0912626.WI': {
            'nc_file': 'AVHRR-GAC_FDR_1C_N06_19810330T113119Z_19810330T120530Z_R_O_20200101T000000Z_0100.nc',
            'midnight_line': np.nan,
            'overlap_free_end': 3222,  # == along_track - 1
            # 3232 (== along_track - 1) in CLARA. The file is just a bit longer there
            # (+4 seconds, 3233 lines. Not sure why...
            'global_quality_flag': QualityFlags.OK
        },
        'NSS.GHRR.NA.D81089.S1221.E1315.B0912627.WI': {
            'nc_file': 'AVHRR-GAC_FDR_1C_N06_19810330T122303Z_19810330T131503Z_R_O_20200101T000000Z_0100.nc',
            'midnight_line': np.nan,
            'overlap_free_end': 6198,  # == along_track -1
            # 6193 (== along_track - 1) in CLARA. The file is just a bit shorter there
            # (-1.5 seconds, 6194 lines in total). Not sure why...
            'global_quality_flag': QualityFlags.OK
        },
        'NSS.GHRR.NA.D81089.S1329.E1459.B0912728.WI': {
            'nc_file': 'AVHRR-GAC_FDR_1C_N06_19810330T132910Z_19810330T145933Z_R_O_20200101T000000Z_0100.nc',
            'midnight_line': np.nan,
            'overlap_free_end': 10770,
            'global_quality_flag': QualityFlags.OK
        },
        'NSS.GHRR.NA.D81089.S1459.E1631.B0912829.GC': {
            'nc_file': 'AVHRR-GAC_FDR_1C_N06_19810330T145914Z_19810330T163142Z_R_O_20200101T000000Z_0100.nc',
            'midnight_line': np.nan,
            'overlap_free_end': 10489,
            'global_quality_flag': QualityFlags.OK
        },
        'NSS.GHRR.NA.D81089.S1626.E1812.B0912930.GC': {
            'nc_file': 'AVHRR-GAC_FDR_1C_N06_19810330T162639Z_19810330T181201Z_R_O_20200101T000000Z_0100.nc',
            'midnight_line': np.nan,
            'overlap_free_end': 12057,
            'global_quality_flag': QualityFlags.OK
        },
        'NSS.GHRR.NA.D81089.S1807.E1951.B0913031.GC': {
            'nc_file': 'AVHRR-GAC_FDR_1C_N06_19810330T180708Z_19810330T195132Z_R_O_20200101T000000Z_0100.nc',
            'midnight_line': np.nan,
            'overlap_free_end': 12011,
            'global_quality_flag': QualityFlags.OK
        },
        'NSS.GHRR.NA.D81089.S1947.E2131.B0913132.GC': {
            'nc_file': 'AVHRR-GAC_FDR_1C_N06_19810330T194718Z_19810330T213158Z_R_O_20200101T000000Z_0100.nc',
            'midnight_line': np.nan,
            'overlap_free_end': 11955,
            'global_quality_flag': QualityFlags.OK
        },
        'NSS.GHRR.NA.D81089.S2126.E2256.B0913233.WI': {
            'nc_file': 'AVHRR-GAC_FDR_1C_N06_19810330T212656Z_19810330T225612Z_R_O_20200101T000000Z_0100.nc',
            'midnight_line': np.nan,
            'overlap_free_end': 10057,
            'global_quality_flag': QualityFlags.OK
        },
        'NSS.GHRR.NA.D81089.S2251.E0035.B0913334.WI': {
            'nc_file': 'AVHRR-GAC_FDR_1C_N06_19810330T225108Z_19810331T003506Z_R_O_20200101T000000Z_0100.nc',
            'midnight_line': 8260,
            'overlap_free_end': 12472,
            # This is the last file here, therefore overlap_free_end == along_track - 1.
            # In CLARA there is overlap with the next file, which is why they have 12101.
            'global_quality_flag': QualityFlags.OK
        },
    }  # from CLARA-A3 Feedback Loop 1

    def test_regression(self):
        self._tst_regression(self.nc_files, self.nc_files_ref)

    def test_mda_clara(self):
        """Test metadata against CLARA-A3 reference."""
        for nc_file in self.nc_files:
            with xr.open_dataset(nc_file) as ds:
                gac_file = ds.attrs['gac_filename']
                mda_exp = self.mda_exp[gac_file].copy()
                mda_exp.pop('nc_file')
                for var_name, exp in mda_exp.items():
                    np.testing.assert_equal(ds[var_name].values, exp)

    def test_cf_compliance(self):
        """Test CF compliance of generated files using CF checker."""
        checker = CFChecker()
        for nc_file in self.nc_files:
            LOG.info('Checking CF compliance of {}'.format(nc_file))
            res = checker.checker(nc_file)
            global_err = any([res['global'][cat] for cat in ('ERROR', 'FATAL')])
            var_err = any([(v['ERROR'] or v['FATAL']) for v in res['variables'].values()])
            err = global_err or var_err
            self.assertFalse(err, msg='{} is not CF compliant'.format(nc_file))


class EndToEndTestCorrupt(EndToEndTestBase):
    """End-to-end test with data that have common defects."""

    tag = 'corrupt'
    with_metadata = False

    def test_regression(self):
        self._tst_regression(self.nc_files, self.nc_files_ref)


if __name__ == '__main__':
    unittest.main(verbosity=2)
