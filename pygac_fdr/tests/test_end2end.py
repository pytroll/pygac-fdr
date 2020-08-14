from cfchecker.cfchecks import CFChecker
import glob
import gzip
import numpy as np
import os
import shutil
import subprocess
import unittest
import xarray as xr

from pygac_fdr.metadata import QualityFlags


class EndToEndTest(unittest.TestCase):
    mda_exp = {
        'AVHRR-GAC_FDR_1C_N06_19810330T005421Z_19810330T024632Z_R_O_20200101T000000Z_0100.nc':
            {'midnight_line': np.nan,
             'overlap_free_end': 12995,
             'global_quality_flag': QualityFlags.OK},
        'AVHRR-GAC_FDR_1C_N06_19810330T024239Z_19810330T042759Z_R_O_20200101T000000Z_0100.nc':
            {'midnight_line': np.nan,
             'overlap_free_end': 12157,
             'global_quality_flag': QualityFlags.OK},
        'AVHRR-GAC_FDR_1C_N06_19810330T042358Z_19810330T060903Z_R_O_20200101T000000Z_0100.nc':
            {'midnight_line': np.nan,
             'overlap_free_end': 12100,
             # 12602 (=along_track-1) in CLARA, but that's because the next file was skipped
             # for some reason
             'global_quality_flag': QualityFlags.OK},
        'AVHRR-GAC_FDR_1C_N06_19810330T060453Z_19810330T075842Z_R_O_20200101T000000Z_0100.nc':
            {'midnight_line': np.nan,
             'overlap_free_end': 13020,  # Skipped in CLARA
             'global_quality_flag': QualityFlags.OK},
        'AVHRR-GAC_FDR_1C_N06_19810330T075339Z_19810330T094742Z_R_O_20200101T000000Z_0100.nc':
            {'midnight_line': np.nan,
             'overlap_free_end': 13053,
             # 13618 (=along_track-1) in CLARA, but that's because only the next file
             # (redundant in this case) is taken into account. It is not taken into
             # account that there might be overlap with the file after that.
             'global_quality_flag': QualityFlags.OK},
        'AVHRR-GAC_FDR_1C_N06_19810330T094300Z_19810330T105846Z_R_O_20200101T000000Z_0100.nc':
            {'midnight_line': np.nan,
             'overlap_free_end': np.nan,
             'global_quality_flag': QualityFlags.REDUNDANT},
        'AVHRR-GAC_FDR_1C_N06_19810330T094300Z_19810330T113615Z_R_O_20200101T000000Z_0100.nc':
            {'midnight_line': np.nan,
             'overlap_free_end': 12989,  # Skipped in CLARA
             'global_quality_flag': QualityFlags.OK},
        'AVHRR-GAC_FDR_1C_N06_19810330T110243Z_19810330T113615Z_R_O_20200101T000000Z_0100.nc':
            {'midnight_line': np.nan,
             'overlap_free_end': np.nan,
             'global_quality_flag': QualityFlags.REDUNDANT},
        'AVHRR-GAC_FDR_1C_N06_19810330T113119Z_19810330T120530Z_R_O_20200101T000000Z_0100.nc':
            {'midnight_line': np.nan,
             'overlap_free_end': 3222,  # == along_track - 1
             # 3232 (== along_track - 1) in CLARA. The file is just a bit longer there
             # (+4 seconds, 3233 lines. Not sure why...
             'global_quality_flag': QualityFlags.OK},
        'AVHRR-GAC_FDR_1C_N06_19810330T122303Z_19810330T131503Z_R_O_20200101T000000Z_0100.nc':
            {'midnight_line': np.nan,
             'overlap_free_end': 6198,  # == along_track -1
             # 6193 (== along_track - 1) in CLARA. The file is just a bit shorter there
             # (-1.5 seconds, 6194 lines in total). Not sure why...
             'global_quality_flag': QualityFlags.OK},
        'AVHRR-GAC_FDR_1C_N06_19810330T132910Z_19810330T145933Z_R_O_20200101T000000Z_0100.nc':
            {'midnight_line': np.nan,
             'overlap_free_end': 10770,
             'global_quality_flag': QualityFlags.OK},
        'AVHRR-GAC_FDR_1C_N06_19810330T145914Z_19810330T163142Z_R_O_20200101T000000Z_0100.nc':
            {'midnight_line': np.nan,
             'overlap_free_end': 10489,
             'global_quality_flag': QualityFlags.OK},
        'AVHRR-GAC_FDR_1C_N06_19810330T162639Z_19810330T181201Z_R_O_20200101T000000Z_0100.nc':
            {'midnight_line': np.nan,
             'overlap_free_end': 12057,
             'global_quality_flag': QualityFlags.OK},
        'AVHRR-GAC_FDR_1C_N06_19810330T180708Z_19810330T195132Z_R_O_20200101T000000Z_0100.nc':
            {'midnight_line': np.nan,
             'overlap_free_end': 12011,
             'global_quality_flag': QualityFlags.OK},
        'AVHRR-GAC_FDR_1C_N06_19810330T194718Z_19810330T213158Z_R_O_20200101T000000Z_0100.nc':
            {'midnight_line': np.nan,
             'overlap_free_end': 11955,
             'global_quality_flag': QualityFlags.OK},
        'AVHRR-GAC_FDR_1C_N06_19810330T212656Z_19810330T225612Z_R_O_20200101T000000Z_0100.nc':
            {'midnight_line': np.nan,
             'overlap_free_end': 10057,
             'global_quality_flag': QualityFlags.OK},
        'AVHRR-GAC_FDR_1C_N06_19810330T225108Z_19810331T003506Z_R_O_20200101T000000Z_0100.nc':
            {'midnight_line': 8260,
             'overlap_free_end': 12472,
             # This is the last file here, therefore overlap_free_end == along_track - 1.
             # In CLARA there is overlap with the next file, which is why they have 12101.
             'global_quality_flag': QualityFlags.OK},
    }  # from CLARA-A3 Feedback Loop 1

    @classmethod
    def _call_subproc(cls, cmd):
        ret = subprocess.call(cmd)
        if ret:
            raise RuntimeError('Subprocess {} failed with return code {}'.format(cmd, ret))

    @classmethod
    def _unzip_gac_files(cls, filenames_gz, output_dir):
        filenames = []
        for filename_gz in filenames_gz:
            basename_gz = os.path.basename(filename_gz)
            basename = os.path.splitext(basename_gz)[0]
            filename = os.path.join(output_dir, os.path.basename(basename))
            with gzip.open(filename_gz, 'rb') as f_in:
                with open(filename, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            filenames.append(filename)
        return filenames

    @classmethod
    def setUpClass(cls):
        cls.fast = os.environ.get('PYGAC_FDR_TEST_FAST', '0') == '1'
        test_data_dir = os.path.join(os.path.dirname(__file__), 'test_data')
        cfg_file = os.path.join(test_data_dir, '../../../etc/pygac-fdr.yaml')
        tle_dir = os.path.join(test_data_dir, 'tle')
        input_dir = os.path.join(test_data_dir, 'input')
        gac_files_gz = sorted(glob.glob(input_dir + '/*.gz'))
        cls.output_dir = os.path.join(test_data_dir, 'output')
        ref_dir = os.path.join(test_data_dir, 'output_ref')
        cls.nc_files_ref = sorted(glob.glob(ref_dir + '/*.nc'))
        dbfile = os.path.join(test_data_dir, 'test.sqlite3')
        if cls.fast:
            gac_files_gz = [gac_files_gz[-1]]
            cls.nc_files_ref = [cls.nc_files_ref[-1]]

        # Prepare output directory
        if not os.path.isdir(cls.output_dir):
            os.makedirs(cls.output_dir)
        cls._cleanup_output_dir()

        # Unzip GAC files
        gac_files = cls._unzip_gac_files(gac_files_gz, cls.output_dir)

        # Run: Read GAC files and write netCDF files
        run = ['pygac-fdr-run',
               '--cfg', cfg_file,
               '--output-dir', cls.output_dir,
               '--tle-dir', tle_dir,
               '--verbose', '--log-all'] + gac_files
        #cls._call_subproc(run)
        cls.nc_files = sorted(glob.glob(cls.output_dir + '/*.nc'))

        # Collect & complement metadata
        collect = ['pygac-fdr-mda-collect', '--dbfile', dbfile, '--if-exists', 'replace',
                   '--verbose'] + cls.nc_files
        #cls._call_subproc(collect)

        # Update metadata
        update = ['pygac-fdr-mda-update', '--dbfile', dbfile]
        #cls._call_subproc(update)

    def test_mda(self):
        for nc_file in self.nc_files:
            try:
                mda_exp = self.mda_exp[os.path.basename(nc_file)]
            except KeyError:
                raise KeyError('No reference metadata for file {}. Did the filename '
                               'timestamps change?'.format(nc_file))
            with xr.open_dataset(nc_file) as ds:
                for var_name, exp in mda_exp.items():
                    np.testing.assert_equal(ds[var_name].values, exp)

    def test_regression(self):
        dynamic_attrs = ['date_created', 'history', 'version_satpy', 'version_pygac',
                         'version_pygac_fdr']
        for nc_file, nc_file_ref in zip(self.nc_files, self.nc_files_ref):
            with xr.open_dataset(nc_file) as ds:
                with xr.open_dataset(nc_file_ref) as ds_ref:
                    # Remove dynamic attributes
                    for attr in dynamic_attrs:
                        ds.attrs.pop(attr)
                        ds_ref.attrs.pop(attr)

                    # If testing just one file, there is no overlap
                    if self.fast:
                        ds = ds.drop_vars(['overlap_free_start',
                                           'overlap_free_end'])
                        ds_ref = ds_ref.drop_vars(['overlap_free_start',
                                                   'overlap_free_end'])

                    # Compare datasets
                    xr.testing.assert_identical(ds, ds_ref)

    def test_cf_compliance(self):
        checker = CFChecker()
        for nc_file in self.nc_files:
            res = checker.checker(nc_file)
            global_err = any([res['global'][cat] for cat in ('ERROR', 'FATAL')])
            var_err = any([(v['ERROR'] or v['FATAL']) for v in res['variables'].values()])
            err = global_err or var_err
            self.assertFalse(err, msg='{} is not CF compliant'.format(nc_file))

    @classmethod
    def _cleanup_output_dir(cls):
        if os.environ.get('PYGAC_FDR_TEST_CLEANUP', '1') == '1':
            for item in os.listdir(cls.output_dir):
                item = os.path.join(cls.output_dir, item)
                if os.path.isfile(item):
                    os.unlink(item)

    @classmethod
    def tearDownClass(cls):
        cls._cleanup_output_dir()
