"""End-to-end tests for pygac-fdr.

Download test data set, run the entire chain of processing steps (read GAC files, write data 
to netCDF, enhance metadata) and compare results against reference data. They should be identical.

Usage:

$ pytest test_end2end.py

The test behaviour can be controlled using the following environment variables (set to 0/1
to disable/enable):

- PYGAC_FDR_TEST_DATA: Where to download test data
- PYGAC_FDR_TEST_FAST: Run tests with only one file
- PYGAC_FDR_TEST_CLEANUP: Cleanup output after testing (successful or not)
- PYGAC_FDR_TEST_RESUME: Resume testing with existing output files instead of generating new ones

"""

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


def assert_data_close_and_attrs_identical(a, b):
    xr.testing.assert_allclose(a, b)
    from xarray.core.utils import dict_equiv
    from xarray.core.formatting import diff_attrs_repr
    assert dict_equiv(a.attrs, b.attrs), diff_attrs_repr(a.attrs, b.attrs, 'identical')


class EndToEndTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fast = os.environ.get('PYGAC_FDR_TEST_FAST', '0') == '1'
        cls.resume = os.environ.get('PYGAC_FDR_TEST_RESUME', '0') == '1'
        cls.test_data_dir = os.environ.get(
            'PYGAC_FDR_TEST_DATA',
            os.path.join(os.path.dirname(__file__), 'test_data')
        )
        if not os.path.isdir(cls.test_data_dir):
            os.makedirs(cls.test_data_dir)
        cls.cfg_file = os.path.join(os.path.dirname(__file__), '../../etc/pygac-fdr.yaml')
        cls.tle_dir = os.path.join(cls.test_data_dir, 'tle')
        cls.input_dir = os.path.join(cls.test_data_dir, 'input')
        cls.output_dir = os.path.join(cls.test_data_dir, 'output')
        cls.output_dir_ref = os.path.join(cls.test_data_dir, 'output_ref')
        cls.fetch_test_data()

    @classmethod
    def _call_subproc(cls, cmd, cwd='.'):
        ret = subprocess.call(cmd, cwd=cwd)
        if ret:
            raise RuntimeError('Subprocess {} failed with return code {}'.format(cmd, ret))

    @classmethod
    def fetch_test_data(cls):
        """Fetch test data (input & reference output).

        Existing files will only be re-downloaded if the server has a newer version.
        """
        cmd = ['wget', '--mirror', '--no-host-directories', '--no-parent', '--cut-dirs=4', '--reject="index.html*"',
               'https://public.cmsaf.dwd.de/data/sfinkens/pygac-fdr/test_data/']
        cls._call_subproc(cmd, cwd=cls.test_data_dir)

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
    def _run(cls, gac_files_gz, dbfile=None):
        if cls.resume:
            # Resume testing with existing output files
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
        cls._call_subproc(run)
        nc_files = sorted(glob.glob(cls.output_dir + '/*.nc'))

        if dbfile:
            # Collect & complement metadata
            collect = ['pygac-fdr-mda-collect', '--dbfile', dbfile, '--if-exists', 'replace',
                       '--verbose'] + nc_files
            cls._call_subproc(collect)

            # Update metadata
            update = ['pygac-fdr-mda-update', '--dbfile', dbfile]
            cls._call_subproc(update)

        return nc_files

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

    def _tst_regression(self, nc_files, nc_files_ref):
        """Test entire netCDF contents against reference files."""
        dynamic_attrs = ['date_created', 'history', 'version_satpy', 'version_pygac',
                         'version_pygac_fdr']
        for nc_file, nc_file_ref in zip(nc_files, nc_files_ref):
            with xr.open_dataset(nc_file) as ds:
                with xr.open_dataset(nc_file_ref) as ds_ref:
                    # Remove dynamic attributes
                    for attr in dynamic_attrs:
                        ds.attrs.pop(attr)
                        ds_ref.attrs.pop(attr)

                    # If testing just one file, there is no overlap
                    if self.fast:
                        ds = ds.drop_vars(['overlap_free_start',
                                           'overlap_free_end'],
                                           errors='ignore')
                        ds_ref = ds_ref.drop_vars(['overlap_free_start',
                                                   'overlap_free_end'],
                                                   errors='ignore')

                    # Compare datasets
                    assert_data_close_and_attrs_identical(ds, ds_ref)


class EndToEndTestNormal(EndToEndTestBase):
    """End-to-end test with normal data (no corruption).

    Also compare metadata against results from CLARA-A3 feedback loop 1.
    """
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

    @classmethod
    def setUpClass(cls):
        super(EndToEndTestNormal, cls).setUpClass()

        gac_files_gz = [os.path.join(cls.input_dir, 'normal', fname + '.gz')
                        for fname in cls.mda_exp.keys()]
        cls.nc_files_ref = [os.path.join(cls.output_dir_ref, 'normal', mda['nc_file'])
                            for mda in cls.mda_exp.values()]
        dbfile = os.path.join(cls.test_data_dir, 'test.sqlite3')
        if cls.fast:
            gac_files_gz = [gac_files_gz[-1]]
            cls.nc_files_ref = [cls.nc_files_ref[-1]]

        # Process "normal" files
        cls.nc_files = cls._run(gac_files_gz, dbfile)

        # Find corresponding reference output
        cls.nc_files_ref = []
        for nc_file in cls.nc_files:
            with xr.open_dataset(nc_file) as ds:
                gac_file = ds.attrs['gac_filename']
                nc_file_ref = os.path.join(cls.output_dir_ref,
                                           'normal',
                                           cls.mda_exp[gac_file]['nc_file'])
                cls.nc_files_ref.append(nc_file_ref)

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
            res = checker.checker(nc_file)
            global_err = any([res['global'][cat] for cat in ('ERROR', 'FATAL')])
            var_err = any([(v['ERROR'] or v['FATAL']) for v in res['variables'].values()])
            err = global_err or var_err
            self.assertFalse(err, msg='{} is not CF compliant'.format(nc_file))


class EndToEndTestCorrupt(EndToEndTestBase):
    """End-to-end test with data that have common defects."""
    @classmethod
    def setUpClass(cls):
        super(EndToEndTestCorrupt, cls).setUpClass()

        gac_files_gz = sorted(
            glob.glob(os.path.join(cls.input_dir, 'corrupt', '*.gz'))
        )
        cls.nc_files_ref = sorted(
            glob.glob(os.path.join(cls.output_dir_ref, 'corrupt', '*.nc'))
        )
        if cls.fast:
            gac_files_gz = [gac_files_gz[-1]]
            cls.nc_files_ref = [cls.nc_files_ref[-1]]

        # Process "corrupt" files
        cls.nc_files = cls._run(gac_files_gz)
        cls.nc_files = sorted(glob.glob(os.path.join(cls.output_dir, '*.nc')))

    def test_regression(self):
        self._tst_regression(self.nc_files, self.nc_files_ref)
