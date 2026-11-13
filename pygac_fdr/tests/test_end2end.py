"""End-to-end regression, metadata and CF-compliance tests.

Run with:

    pytest -vs test_end2end.py
    pytest -vs test_end2end.py::test_regression[normal]

Configure test data, tolerances and plotting in test_end2end.yaml.
"""

import gzip
import logging
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr
import yaml
from cfchecker.cfchecks import CFChecker
from click.testing import CliRunner
from dateutil.parser import isoparse
from pytest_subtests import SubTests
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
        "equator_crossing_longitude": np.array([-79.094, -104.398]),
        "equator_crossing_time": np.array(
            ["1981-03-30T00:55:43.706000000", "1981-03-30T02:36:57.706000000"],
            dtype="datetime64[ns]",
        ),
    },
    "NSS.GHRR.NA.D81089.S0242.E0427.B0912122.GC": {
        "midnight_line": np.nan,
        "overlap_free_end": 12157,
        "global_quality_flag": QualityFlags.OK,
        "equator_crossing_longitude": np.array([-129.711, -129.711]),
        "equator_crossing_time": np.array(
            ["1981-03-30T04:18:11.706000000", "1981-03-30T04:18:12.206000000"],
            dtype="datetime64[ns]",
        ),
    },
    "NSS.GHRR.NA.D81089.S0423.E0609.B0912223.GC": {
        "midnight_line": np.nan,
        "overlap_free_end": 12603,
        "global_quality_flag": QualityFlags.OK,
        "equator_crossing_longitude": np.array([-155.016, np.nan]),
        "equator_crossing_time": np.array(
            ["1981-03-30T05:59:25.706000000", "NaT"],
            dtype="datetime64[ns]",
        ),
    },
    "NSS.GHRR.NA.D81089.S0753.E0947.B0912425.WI": {
        "midnight_line": np.nan,
        "overlap_free_end": 13053,
        "global_quality_flag": QualityFlags.OK,
        "equator_crossing_longitude": np.array([154.359, np.nan]),
        "equator_crossing_time": np.array(
            ["1981-03-30T09:21:53.706000000", "NaT"],
            dtype="datetime64[ns]",
        ),
    },
    "NSS.GHRR.NA.D81089.S0943.E1058.B0912525.WI": {
        "midnight_line": np.nan,
        "overlap_free_end": 9043,
        "global_quality_flag": QualityFlags.OK,
        "equator_crossing_longitude": np.array([np.nan, np.nan]),
        "equator_crossing_time": np.array(["NaT", "NaT"], dtype="datetime64[ns]"),
    },
    "NSS.GHRR.NA.D81089.S1102.E1136.B0912626.WI": {
        "midnight_line": np.nan,
        "overlap_free_end": 3408,
        "global_quality_flag": QualityFlags.OK,
        "equator_crossing_longitude": np.array([129.055, np.nan]),
        "equator_crossing_time": np.array(
            ["1981-03-30T11:03:07.206000000", "NaT"],
            dtype="datetime64[ns]",
        ),
    },
    "NSS.GHRR.NA.D81089.S1131.E1205.B0912626.WI": {
        "midnight_line": np.nan,
        "overlap_free_end": 3232,
        "global_quality_flag": QualityFlags.OK,
        "equator_crossing_longitude": np.array([-97.32, -67.008]),
        "equator_crossing_time": np.array(
            ["1981-03-30T11:39:40.206000000", "1981-03-30T11:40:26.706000000"],
            dtype="datetime64[ns]",
        ),
    },
    "NSS.GHRR.NA.D81089.S1221.E1315.B0912627.WI": {
        "midnight_line": np.nan,
        "overlap_free_end": 6194,
        "global_quality_flag": QualityFlags.OK,
        "equator_crossing_longitude": np.array([130.703, 103.742]),
        "equator_crossing_time": np.array(
            ["1981-03-30T12:24:46.206000000", "1981-03-30T12:44:21.206000000"],
            dtype="datetime64[ns]",
        ),
    },
    "NSS.GHRR.NA.D81089.S1329.E1459.B0912728.WI": {
        "midnight_line": np.nan,
        "overlap_free_end": 10770,
        "global_quality_flag": QualityFlags.OK,
        "equator_crossing_longitude": np.array([78.43, np.nan]),
        "equator_crossing_time": np.array(
            ["1981-03-30T14:25:35.206000000", "NaT"],
            dtype="datetime64[ns]",
        ),
    },
    "NSS.GHRR.NA.D81089.S1459.E1631.B0912829.GC": {
        "midnight_line": np.nan,
        "overlap_free_end": 10489,
        "global_quality_flag": QualityFlags.OK,
        "equator_crossing_longitude": np.array([53.125, np.nan]),
        "equator_crossing_time": np.array(
            ["1981-03-30T16:06:49.206000000", "NaT"],
            dtype="datetime64[ns]",
        ),
    },
    "NSS.GHRR.NA.D81089.S1626.E1812.B0912930.GC": {
        "midnight_line": np.nan,
        "overlap_free_end": 12057,
        "global_quality_flag": QualityFlags.OK,
        "equator_crossing_longitude": np.array([27.82, np.nan]),
        "equator_crossing_time": np.array(
            ["1981-03-30T17:48:02.706000000", "NaT"],
            dtype="datetime64[ns]",
        ),
    },
    "NSS.GHRR.NA.D81089.S1807.E1951.B0913031.GC": {
        "midnight_line": np.nan,
        "overlap_free_end": 12011,
        "global_quality_flag": QualityFlags.OK,
        "equator_crossing_longitude": np.array([2.516, np.nan]),
        "equator_crossing_time": np.array(
            ["1981-03-30T19:29:16.706000000", "NaT"],
            dtype="datetime64[ns]",
        ),
    },
    "NSS.GHRR.NA.D81089.S1947.E2131.B0913132.GC": {
        "midnight_line": np.nan,
        "overlap_free_end": 11955,
        "global_quality_flag": QualityFlags.OK,
        "equator_crossing_longitude": np.array([-22.789, np.nan]),
        "equator_crossing_time": np.array(
            ["1981-03-30T21:10:30.706000000", "NaT"],
            dtype="datetime64[ns]",
        ),
    },
    "NSS.GHRR.NA.D81089.S2126.E2256.B0913233.WI": {
        "midnight_line": np.nan,
        "overlap_free_end": 10057,
        "global_quality_flag": QualityFlags.OK,
        "equator_crossing_longitude": np.array([-48.094, np.nan]),
        "equator_crossing_time": np.array(
            ["1981-03-30T22:51:44.206000000", "NaT"],
            dtype="datetime64[ns]",
        ),
    },
    "NSS.GHRR.NA.D81089.S2251.E0035.B0913334.WI": {
        "midnight_line": 8260,
        "overlap_free_end": 12473,
        # This is the last file here, therefore overlap_free_end == along_track - 1.
        # In CLARA there is overlap with the next file, which is why they have 12101.
        "global_quality_flag": QualityFlags.OK,
        "equator_crossing_longitude": np.array([-48.094, -73.398]),
        "equator_crossing_time": np.array(
            ["1981-03-30T22:51:44.206000000", "1981-03-31T00:32:57.706000000"],
            dtype="datetime64[ns]",
        ),
    },
}  # from CLARA-A3 Feedback Loop 1


FAST_FILES = {
    "normal": {
        "l1b": "NSS.GHRR.NA.D81089.S0054.E0246.B0912021.GC.gz",
        "l1c": "AVHRR-GAC_FDR_1C_N06_19810330T005421Z_19810330T024632Z_R_O_20200101T000000Z_0100.nc",
    },
    "corrupt": {
        "l1b": "NSS.GHRR.M1.D15362.S0031.E0129.B1699697.SV.gz",
        "l1c": "AVHRR-GAC_FDR_1C_M01_20151228T003207Z_20151228T012926Z_R_O_20200101T000000Z_0100.nc",
    },
}


@dataclass
class Config:
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
    plot: bool
    trigger_failure: bool


def _paths_to_strings(file_paths: list[Path]) -> list[str]:
    return list(map(str, file_paths))


def _call_cli(command: list[str]) -> None:
    result = CliRunner().invoke(cli, command, catch_exceptions=False)
    assert result.exit_code == 0, result.output


@pytest.fixture(scope="module")
def config() -> Config:
    logging_on(logging.DEBUG, for_all=True)

    with open(Path(__file__).with_name("test_end2end.yaml")) as fh:
        config_data = yaml.safe_load(fh)

    data_dir = Path(config_data["test_data_dir"])
    config = Config(
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
        plot=config_data["plot"],
        trigger_failure=config_data["trigger_failure"],
    )
    if config.fast:
        LOG.info("Fast mode, processing just one file.")
    return config


def _cleanup_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        for item in output_dir.iterdir():
            if item.is_file():
                item.unlink()


def _unzip_gac_files(filenames_gz: list[Path], output_dir: Path) -> list[Path]:
    unzipped = []
    for file_gz in filenames_gz:
        file_unzip = output_dir / file_gz.stem
        LOG.info("Decompressing %s", file_gz)
        with gzip.open(file_gz, "rb") as source:
            with open(file_unzip, "wb") as destination:
                shutil.copyfileobj(source, destination)
        unzipped.append(file_unzip)
    return unzipped


def _generate_l1c_files(
    config: Config, variant: str, *, with_metadata: bool
) -> list[Path]:
    input_dir = config.input_dir / variant
    output_dir = config.output_dir / variant

    if config.resume:
        LOG.info("Resuming %s tests with existing output files", variant)
        return _collect_l1c_files_from_previous_run(output_dir, variant, config.fast)

    _make_fresh_output_dir(output_dir)
    l1b_files_gz = _collect_l1b_gz_files(input_dir, variant, config.fast)
    l1b_files = _unzip_gac_files(l1b_files_gz, output_dir)
    l1c_files = _run_pygac_fdr(l1b_files, config.cfg_file, config.tle_dir, output_dir)
    if with_metadata:
        _update_metadata(l1c_files, output_dir)
    return l1c_files


def _make_fresh_output_dir(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_output_dir(output_dir)


def _collect_l1c_files_from_previous_run(
    output_dir: Path, variant: str, fast: bool
) -> list[Path]:
    if fast:
        filename = output_dir / FAST_FILES[variant]["l1c"]
        l1c_files = [filename] if filename.is_file() else []
    else:
        l1c_files = sorted(output_dir.glob("*.nc"))
    if not l1c_files:
        raise RuntimeError("No l1c files from previous run")
    return l1c_files


def _collect_l1b_gz_files(input_dir: Path, variant: str, fast: bool) -> list[Path]:
    if fast:
        return [input_dir / FAST_FILES[variant]["l1b"]]
    return sorted(input_dir.glob("*.gz"))


def _run_pygac_fdr(
    l1b_files: list[Path], cfg_file: Path, tle_dir: Path, output_dir: Path
) -> list[Path]:
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


def _update_metadata(l1c_files: list[Path], output_dir: Path) -> None:
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
def l1c_files_normal(config: Config) -> list[Path]:
    return _generate_l1c_files(config, "normal", with_metadata=True)


@pytest.fixture(scope="module")
def l1c_files_corrupt(config: Config) -> list[Path]:
    return _generate_l1c_files(config, "corrupt", with_metadata=False)


@pytest.fixture
def l1c_files(request: pytest.FixtureRequest, variant: str) -> list[Path]:
    return request.getfixturevalue(f"l1c_files_{variant}")


@pytest.fixture
def l1c_files_exp(variant: str, config: Config) -> list[Path]:
    ref_dir = config.output_ref_dir / variant
    if config.fast:
        return [ref_dir / FAST_FILES[variant]["l1c"]]
    return sorted(ref_dir.glob("*.nc"))


@pytest.mark.parametrize("variant", ["normal", "corrupt"])
def test_l1c_files(l1c_files: list[Path], l1c_files_exp: list[Path]) -> None:
    """Test that all l1c files were generated."""
    names_expected = [f.name for f in sorted(l1c_files_exp)]
    names_actual = [f.name for f in sorted(l1c_files)]
    assert names_expected == names_actual


@pytest.mark.parametrize("variant", ["normal", "corrupt"])
def test_regression(
    variant: str,
    l1c_files: list[Path],
    config: Config,
    subtests: SubTests,
) -> None:
    """Test that l1c files equal the reference."""
    for output_file in l1c_files:
        with subtests.test(file=output_file.name):
            _compare_to_reference(output_file, variant, config)


def _compare_to_reference(output_file: Path, variant: str, config: Config) -> None:
    ds_tst, ds_ref = _read_file_pair(output_file, variant, config.output_ref_dir)
    ds_tst, ds_ref = _prepare_datasets_for_comparison(ds_tst, ds_ref, config)
    plot_on_failure = _plot_diffs_on_failure(
        ds_tst, ds_ref, output_file, enabled=config.plot
    )
    with plot_on_failure:
        _assert_datasets_close(ds_tst, ds_ref, config)
    _assert_global_attrs_close(ds_tst.attrs, ds_ref.attrs, config)
    _assert_variable_attrs_equal(ds_tst, ds_ref)


def _read_file_pair(
    output_file: Path, variant: str, ref_dir: Path
) -> tuple[xr.Dataset, xr.Dataset]:
    reference_file = ref_dir / variant / output_file.name
    with (
        xr.open_dataset(output_file) as ds_tst,
        xr.open_dataset(reference_file) as ds_ref,
    ):
        return ds_tst, ds_ref


def _prepare_datasets_for_comparison(
    ds_tst: xr.Dataset, ds_ref: xr.Dataset, config: Config
) -> tuple[xr.Dataset, xr.Dataset]:
    for attr in DYNAMIC_ATTRS:
        ds_tst.attrs.pop(attr)
        ds_ref.attrs.pop(attr)

    if config.fast:
        # There's no overlap with a single file
        exclusions = ["overlap_free_start", "overlap_free_end"]
        ds_tst = ds_tst.drop_vars(exclusions, errors="ignore")
        ds_ref = ds_ref.drop_vars(exclusions, errors="ignore")

    if config.trigger_failure:
        ds_tst["latitude"] = ds_tst["latitude"] * 2
        ds_tst.attrs["geospatial_lat_min"] = 9999.0

    return ds_tst, ds_ref


def _assert_datasets_close(
    ds_tst: xr.Dataset, ds_ref: xr.Dataset, config: Config
) -> None:
    xr.testing.assert_allclose(
        ds_tst,
        ds_ref,
        atol=config.atol,
        rtol=config.rtol,
    )


def _assert_global_attrs_close(
    actual: dict[str, Any], expected: dict[str, Any], config: Config
) -> None:
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


def _assert_variable_attrs_equal(actual: xr.Dataset, expected: xr.Dataset) -> None:
    assert set(actual.variables) == set(expected.variables)
    for name in actual.variables:
        actual_attrs = actual[name].attrs
        expected_attrs = expected[name].attrs
        assert dict_equiv(actual_attrs, expected_attrs), f"{name}:\n" + diff_attrs_repr(
            actual_attrs, expected_attrs, "identical"
        )


@contextmanager
def _plot_diffs_on_failure(
    ds_tst: xr.Dataset,
    ds_ref: xr.Dataset,
    output_file: Path,
    *,
    enabled: bool,
) -> Iterator[None]:
    try:
        yield
    except AssertionError:
        if enabled:
            LOG.info("Plotting diffs")
            _plot_diffs(ds_tst, ds_ref, output_file)
        raise


def _plot_diffs(ds_tst: xr.Dataset, ds_ref: xr.Dataset, output_file: Path) -> None:
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


def test_metadata_clara(
    l1c_files_normal: list[Path], subtests: SubTests, config: Config
) -> None:
    """Compare metadata (normal variant) against CLARA-A3 expectations."""
    if config.fast:
        # No overlap when testing just a single file
        pytest.skip()
    comp_methods = {"equator_crossing_longitude": np.testing.assert_allclose}
    for output_file in l1c_files_normal:
        with subtests.test(file=output_file.name):
            with xr.open_dataset(output_file) as ds:
                l1b_file = ds.attrs["gac_filename"]
                expected = CLARA_EXPECTATIONS[l1b_file]
                for var_name, exp_value in expected.items():
                    comp_method = comp_methods.get(var_name, np.testing.assert_equal)
                    comp_method(
                        ds[var_name].values,
                        exp_value,
                        err_msg=f"{output_file.name}: {var_name}",
                    )


def test_cf_compliance(l1c_files_normal: list[Path], subtests: SubTests) -> None:
    """Check that l1c files (normal variant) are CF compliant."""
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
