"""End-to-end regression, metadata and CF-compliance tests.

Run with:

    pytest -vs test_end2end.py
    pytest -vs test_end2end.py::test_l1c_files_equal_reference[normal]

Configure test data, cleanup, tolerances and plotting in test_end2end.yaml.
"""

import gzip
import logging
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import yaml
from cfchecker.cfchecks import CFChecker
from click.testing import CliRunner
from dateutil.parser import isoparse
from xarray.core.formatting import diff_attrs_repr
from xarray.core.utils import dict_equiv

from pygac_fdr.metadata import QualityFlags
from pygac_fdr.scripts.cli import cli
from pygac_fdr.utils import LOGGER_NAME, logging_on

LOG = logging.getLogger(LOGGER_NAME)

DYNAMIC_ATTRS = (
    "date_created",
    "history",
    "version_satpy",
    "version_pygac",
    "version_pygac_fdr",
)
TIME_ATTRS = ("start_time", "end_time")
NUMERICAL_ATTRS = (
    "geospatial_lon_min",
    "geospatial_lon_max",
    "geospatial_lat_min",
    "geospatial_lat_max",
)

CLARA_EXPECTATIONS = {
    "NSS.GHRR.NA.D81089.S0054.E0246.B0912021.GC": {
        "midnight_line": np.nan,
        "overlap_free_end": 12995,
        "global_quality_flag": QualityFlags.OK,
    },
    "NSS.GHRR.NA.D81089.S0242.E0427.B0912122.GC": {
        "midnight_line": np.nan,
        "overlap_free_end": 12157,
        "global_quality_flag": QualityFlags.OK,
    },
    "NSS.GHRR.NA.D81089.S0423.E0609.B0912223.GC": {
        "midnight_line": np.nan,
        "overlap_free_end": 12603,
        "global_quality_flag": QualityFlags.OK,
    },
    "NSS.GHRR.NA.D81089.S0753.E0947.B0912425.WI": {
        "midnight_line": np.nan,
        "overlap_free_end": 13053,
        "global_quality_flag": QualityFlags.OK,
    },
    "NSS.GHRR.NA.D81089.S0943.E1058.B0912525.WI": {
        "midnight_line": np.nan,
        "overlap_free_end": 9043,
        "global_quality_flag": QualityFlags.OK,
    },
    "NSS.GHRR.NA.D81089.S1102.E1136.B0912626.WI": {
        "midnight_line": np.nan,
        "overlap_free_end": 3408,
        "global_quality_flag": QualityFlags.OK,
    },
    "NSS.GHRR.NA.D81089.S1131.E1205.B0912626.WI": {
        "midnight_line": np.nan,
        "overlap_free_end": 3232,
        "global_quality_flag": QualityFlags.OK,
    },
    "NSS.GHRR.NA.D81089.S1221.E1315.B0912627.WI": {
        "midnight_line": np.nan,
        "overlap_free_end": 6194,
        "global_quality_flag": QualityFlags.OK,
    },
    "NSS.GHRR.NA.D81089.S1329.E1459.B0912728.WI": {
        "midnight_line": np.nan,
        "overlap_free_end": 10770,
        "global_quality_flag": QualityFlags.OK,
    },
    "NSS.GHRR.NA.D81089.S1459.E1631.B0912829.GC": {
        "midnight_line": np.nan,
        "overlap_free_end": 10489,
        "global_quality_flag": QualityFlags.OK,
    },
    "NSS.GHRR.NA.D81089.S1626.E1812.B0912930.GC": {
        "midnight_line": np.nan,
        "overlap_free_end": 12057,
        "global_quality_flag": QualityFlags.OK,
    },
    "NSS.GHRR.NA.D81089.S1807.E1951.B0913031.GC": {
        "midnight_line": np.nan,
        "overlap_free_end": 12011,
        "global_quality_flag": QualityFlags.OK,
    },
    "NSS.GHRR.NA.D81089.S1947.E2131.B0913132.GC": {
        "midnight_line": np.nan,
        "overlap_free_end": 11955,
        "global_quality_flag": QualityFlags.OK,
    },
    "NSS.GHRR.NA.D81089.S2126.E2256.B0913233.WI": {
        "midnight_line": np.nan,
        "overlap_free_end": 10057,
        "global_quality_flag": QualityFlags.OK,
    },
    "NSS.GHRR.NA.D81089.S2251.E0035.B0913334.WI": {
        "midnight_line": 8260,
        "overlap_free_end": 12473,
        # This is the last file here, therefore overlap_free_end == along_track - 1.
        # In CLARA there is overlap with the next file, which is why they have 12101.
        "global_quality_flag": QualityFlags.OK,
    },
}  # from CLARA-A3 Feedback Loop 1


@dataclass
class EndToEndConfig:
    test_data_dir: Path
    input_dir: Path
    output_dir: Path
    output_ref_dir: Path
    tle_dir: Path
    cfg_file: Path
    rtol: float
    atol: float
    fast: bool
    resume: bool
    cleanup: bool
    plot: bool
    trigger_failure: bool


def _paths_to_strings(file_paths: list[Path]) -> list[str]:
    return list(map(str, file_paths))


def _call_cli(command):
    result = CliRunner().invoke(cli, command, catch_exceptions=False)
    assert result.exit_code == 0, result.output


@pytest.fixture(scope="module")
def test_config():
    logging_on(logging.DEBUG, for_all=True)

    with open(Path(__file__).with_name("test_end2end.yaml")) as fh:
        config_data = yaml.safe_load(fh)

    data_dir = Path(config_data["test_data_dir"])
    config = EndToEndConfig(
        test_data_dir=data_dir,
        input_dir=data_dir / "input",
        output_dir=data_dir / "output",
        output_ref_dir=data_dir / "output_ref",
        tle_dir=data_dir / "tle",
        cfg_file=Path(__file__).resolve().parents[2] / "etc" / "pygac-fdr.yaml",
        rtol=float(config_data["rtol"]),
        atol=float(config_data["atol"]),
        fast=config_data["fast"],
        resume=config_data["resume"],
        cleanup=config_data["cleanup"],
        plot=config_data["plot"],
        trigger_failure=config_data["trigger_failure"],
    )
    if config.fast:
        LOG.info("Fast mode, processing just one file.")
    return config


def _cleanup_output_dir(output_dir, *, enabled):
    if enabled and output_dir.exists():
        for item in output_dir.iterdir():
            if item.is_file():
                item.unlink()


def _unzip_gac_files(filenames_gz, output_dir):
    filenames = []
    for filename_gz in filenames_gz:
        filename = output_dir / filename_gz.stem
        LOG.info("Decompressing %s", filename_gz)
        with gzip.open(filename_gz, "rb") as source:
            with open(filename, "wb") as destination:
                shutil.copyfileobj(source, destination)
        filenames.append(filename)
    return filenames


def _generate_l1c_files(config, variant, *, with_metadata):
    input_dir = config.input_dir / variant
    output_dir = config.output_dir / variant

    if config.resume:
        LOG.info("Resuming %s tests with existing output files", variant)
        return _collect_l1c_files_from_previous_run(output_dir, config.fast)

    output_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_output_dir(output_dir, enabled=config.cleanup)

    l1b_files_gz = _collect_l1b_gz_files(input_dir, config.fast)
    l1b_files = _unzip_gac_files(l1b_files_gz, output_dir)
    l1c_files = _run_pygac_fdr(l1b_files, config.cfg_file, config.tle_dir, output_dir)
    if with_metadata:
        _update_metadata(l1c_files, output_dir)
    return l1c_files


def _collect_l1c_files_from_previous_run(output_dir, fast):
    l1c_files = sorted(output_dir.glob("*.nc"))
    if fast:
        l1c_files = l1c_files[-1:]
    if not l1c_files:
        raise RuntimeError("No l1c files from previous run")
    return l1c_files


def _collect_l1b_gz_files(input_dir, fast):
    l1b_files_gz = sorted(input_dir.glob("*.gz"))
    if fast:
        return l1b_files_gz[-1:]
    return l1b_files_gz


def _run_pygac_fdr(l1b_files, cfg_file, tle_dir, output_dir):
    cmd = [
        "run",
        "--cfg",
        str(cfg_file),
        "--output-dir",
        str(output_dir),
        "--tle-dir",
        str(tle_dir),
        "--verbose",
        "--log-all",
    ] + _paths_to_strings(l1b_files)
    _call_cli(cmd)
    return sorted(output_dir.glob("*.nc"))


def _update_metadata(l1c_files, output_dir):
    dbfile = output_dir / "test.sqlite3"
    collect = [
        "mda",
        "collect",
        "--dbfile",
        str(dbfile),
        "--if-exists",
        "replace",
        "--verbose",
    ] + _paths_to_strings(l1c_files)
    _call_cli(collect)
    _call_cli(["mda", "update", "--dbfile", str(dbfile)])


@pytest.fixture(scope="module")
def l1c_files_normal(test_config):
    try:
        yield _generate_l1c_files(test_config, "normal", with_metadata=True)
    finally:
        _cleanup_output_dir(
            test_config.output_dir / "normal",
            enabled=test_config.cleanup,
        )


@pytest.fixture(scope="module")
def l1c_files_corrupt(test_config):
    try:
        yield _generate_l1c_files(test_config, "corrupt", with_metadata=False)
    finally:
        _cleanup_output_dir(
            test_config.output_dir / "corrupt",
            enabled=test_config.cleanup,
        )


@pytest.fixture
def l1c_files(request, variant):
    return request.getfixturevalue(f"l1c_files_{variant}")


@pytest.mark.parametrize("variant", ["normal", "corrupt"])
def test_l1c_files_generated(variant, l1c_files, test_config):
    ref_dir = test_config.output_ref_dir / variant
    names_exp = [f.name for f in sorted(ref_dir.glob("*.nc"))]
    names_actual = [f.name for f in sorted(l1c_files)]
    assert names_exp == names_actual


@pytest.mark.parametrize("variant", ["normal", "corrupt"])
def test_l1c_files_equal_reference(variant, l1c_files, test_config, subtests):
    config = test_config
    for output_file in l1c_files:
        with subtests.test(file=output_file.name):
            _compare_to_reference(output_file, variant, config)


def _compare_to_reference(output_file, variant, config):
    ds_tst, ds_ref = _read_file_pair(output_file, variant, config.output_ref_dir)
    ds_tst, ds_ref = _prepare_datasets_for_comparison(ds_tst, ds_ref, config)
    plot_on_failure = _plot_diffs_on_failure(
        ds_tst, ds_ref, output_file, enabled=config.plot
    )
    with plot_on_failure:
        _assert_datasets_close(ds_tst, ds_ref, config)
    _assert_global_attrs_close(ds_tst.attrs, ds_ref.attrs, config)
    _assert_variable_attrs_equal(ds_tst, ds_ref)


def _read_file_pair(output_file, variant, ref_dir):
    reference_file = ref_dir / variant / output_file.name
    with (
        xr.open_dataset(output_file) as ds_tst,
        xr.open_dataset(reference_file) as ds_ref,
    ):
        return ds_tst, ds_ref


def _prepare_datasets_for_comparison(ds_tst, ds_ref, test_config):
    for attr in DYNAMIC_ATTRS:
        ds_tst.attrs.pop(attr)
        ds_ref.attrs.pop(attr)

    if test_config.fast:
        # There's no overlap with a single file
        exclusions = ["overlap_free_start", "overlap_free_end"]
        ds_tst = ds_tst.drop_vars(exclusions, errors="ignore")
        ds_ref = ds_ref.drop_vars(exclusions, errors="ignore")

    if test_config.trigger_failure:
        ds_tst["latitude"] = ds_tst["latitude"] * 2
        ds_tst.attrs["geospatial_lat_min"] = 9999.0

    return ds_tst, ds_ref


def _assert_datasets_close(ds_tst, ds_ref, test_config):
    xr.testing.assert_allclose(
        ds_tst,
        ds_ref,
        atol=test_config.atol,
        rtol=test_config.rtol,
    )


def _assert_global_attrs_close(actual, expected, config):
    actual = actual.copy()
    expected = expected.copy()

    for attr in TIME_ATTRS:
        actual_time = isoparse(actual.pop(attr))
        expected_time = isoparse(expected.pop(attr))
        assert abs((expected_time - actual_time).total_seconds()) < 2, attr

    for attr in NUMERICAL_ATTRS:
        np.testing.assert_allclose(
            actual.pop(attr),
            expected.pop(attr),
            rtol=config.rtol,
            atol=config.atol,
            err_msg=attr,
        )

    assert dict_equiv(actual, expected), diff_attrs_repr(actual, expected, "identical")


def _assert_variable_attrs_equal(actual, expected):
    assert set(actual.variables) == set(expected.variables)
    for name in actual.variables:
        actual_attrs = actual[name].attrs
        expected_attrs = expected[name].attrs
        assert dict_equiv(actual_attrs, expected_attrs), f"{name}:\n" + diff_attrs_repr(
            actual_attrs, expected_attrs, "identical"
        )


@contextmanager
def _plot_diffs_on_failure(ds_tst, ds_ref, output_file, *, enabled):
    try:
        yield
    except AssertionError:
        if enabled:
            LOG.info("Plotting diffs")
            _plot_diffs(ds_tst, ds_ref, output_file)
        raise


def _plot_diffs(ds_tst, ds_ref, output_file):
    """Save difference plots."""
    import matplotlib.pyplot as plt

    variables = {
        "reflectance_channel_1": {},
        "reflectance_channel_2": {},
        "brightness_temperature_channel_3": {},
        "reflectance_channel_3a": {},
        "brightness_temperature_channel_3b": {},
        "brightness_temperature_channel_4": {},
        "brightness_temperature_channel_5": {},
        "longitude": {
            "vmin": -180,
            "vmax": 180,
            "vmin_diff": -0.1,
            "vmax_diff": 0.1,
        },
        "latitude": {
            "vmin": -90,
            "vmax": 90,
            "vmin_diff": -0.1,
            "vmax_diff": 0.1,
        },
        "solar_zenith_angle": {},
        "solar_azimuth_angle": {},
        "sensor_zenith_angle": {},
        "sensor_azimuth_angle": {},
        "sun_sensor_azimuth_difference_angle": {},
    }

    for name, properties in variables.items():
        try:
            reference = ds_ref[name]
            actual = ds_tst[name]
        except KeyError:
            continue
        difference = actual - reference
        vmin = min(actual.min().item(), reference.min().item())
        vmax = max(actual.max().item(), reference.max().item())
        abs_max_diff = max(
            abs(difference.min().item()),
            abs(difference.max().item()),
        )

        fig, (ax_ref, ax_tst, ax_diff) = plt.subplots(
            nrows=3, figsize=(20, 8.5), sharex=True
        )
        try:
            for data, axis, title in (
                (reference, ax_ref, "Reference"),
                (actual, ax_tst, "Test"),
            ):
                data.transpose().plot.imshow(
                    ax=axis,
                    vmin=properties.get("vmin", vmin),
                    vmax=properties.get("vmax", vmax),
                )
                axis.set_title(title, fontsize=10)
                axis.set_xlabel(None)

            difference.transpose().plot.imshow(
                ax=ax_diff,
                vmin=properties.get("vmin_diff", -0.8 * abs_max_diff),
                vmax=properties.get("vmax_diff", 0.8 * abs_max_diff),
                cmap="RdBu_r",
            )
            ax_diff.set_title("Test - Reference", fontsize=10)
            fig.suptitle(output_file.name)
            fig.savefig(
                output_file.with_name(f"{output_file.name}_{name}.png"),
                bbox_inches="tight",
            )
        finally:
            plt.close(fig)


def test_l1c_metadata_clara(l1c_files_normal, subtests):
    """Compare normal-file metadata against CLARA-A3 expectations."""
    for output_file in l1c_files_normal:
        with subtests.test(file=output_file.name):
            with xr.open_dataset(output_file) as ds:
                l1b_file = ds.attrs["gac_filename"]
                expected = CLARA_EXPECTATIONS[l1b_file]
                for name, value in expected.items():
                    np.testing.assert_equal(
                        ds[name].values,
                        value,
                        err_msg=f"{output_file.name}: {name}",
                    )


def test_l1c_files_are_cf_compliant(l1c_files_normal, subtests):
    """Check generated normal files for CF errors and fatal errors."""
    checker = CFChecker()
    for output_file in l1c_files_normal:
        with subtests.test(file=output_file.name):
            result = checker.checker(str(output_file))
            global_errors = any(
                result["global"][category] for category in ("ERROR", "FATAL")
            )
            variable_errors = any(
                variable["ERROR"] or variable["FATAL"]
                for variable in result["variables"].values()
            )
            assert not (
                global_errors or variable_errors
            ), f"{output_file} is not CF compliant"
