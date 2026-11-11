"""pygac-fdr command line interface."""

import logging
import pathlib
import re
import tarfile
from contextlib import closing

import click
import pandas as pd
import satpy
from dateutil.parser import isoparse
from satpy.readers.core.remote import FSFile

import pygac_fdr.validation as validation
from pygac_fdr.config import read_config
from pygac_fdr.crop import CROP_OVERLAP_BEGINNING, CROP_OVERLAP_END, crop
from pygac_fdr.metadata import (
    MetadataCollector,
    MetadataEnhancer,
    MetadataUpdater,
    read_metadata_from_database,
    save_metadata_to_database,
)
from pygac_fdr.reader import read_gac
from pygac_fdr.utils import LOGGER_NAME, TarFileSystem, logging_on
from pygac_fdr.writer import NetcdfWriter

LOG = logging.getLogger(LOGGER_NAME)


class ArgumentsFileCommand(click.Command):
    """Expand @files as argparse's fromfile_prefix_chars does."""

    def parse_args(self, ctx, args):
        def expand(arguments):
            result = []
            for argument in arguments:
                if argument.startswith("@"):
                    filename = argument[1:]
                    try:
                        with open(filename) as source:
                            result.extend(expand(source.read().splitlines()))
                    except OSError as exc:
                        raise click.UsageError(str(exc), ctx) from exc
                else:
                    result.append(argument)
            return result

        return super().parse_args(ctx, expand(args))


class HideGzFSFile(FSFile):
    _gz_suffix = re.compile(r"\.gz$")

    def __fspath__(self):
        path = super().__fspath__()
        return self._gz_suffix.sub("", path)


def process_file(filename, config):
    LOG.info("Processing file {}".format(filename))
    try:
        scene = read_gac(
            filename, reader_kwargs=config["controls"].get("reader_kwargs")
        )
        writer = NetcdfWriter(
            output_dir=config["output"].get("output_dir"),
            global_attrs=config.get("global_attrs"),
            gac_header_attrs=config.get("gac_header_attrs"),
            fname_fmt=config["output"].get("fname_fmt"),
            encoding=config["netcdf"].get("encoding"),
            engine=config["netcdf"].get("engine"),
            debug=config["controls"].get("debug"),
        )
        writer.write(scene=scene)
        success = True
    except Exception as err:
        if config["controls"]["debug"]:
            raise
        LOG.error("Error processing file {}: {}".format(filename, err))
        success = False
    return success


def process_tarball(tarball, config):
    LOG.info("Processing tarball {}".format(tarball))
    failures = []
    with closing(TarFileSystem(tarball)) as archive:
        filepaths = archive.find("", details=False, withdirs=False)
        for filepath in filepaths:
            filename = HideGzFSFile(filepath, fs=archive)
            success = process_file(filename, config)
            if not success:
                failures.append(filename)
    if failures:
        LOG.error("Could not process the following files: {}".format(failures))
    else:
        LOG.info("Successfully processed all files in {}".format(tarball))


@click.group()
def cli():
    """Process and validate pygac-fdr data."""


@cli.command()
@click.option("--cfg", required=True, help="Path to pygac-fdr configuration file.")
@click.option(
    "--output-dir", help="Override the output directory in the configuration."
)
@click.option("--tle-dir", help="Override the TLE directory in the configuration.")
@click.option("--debug", is_flag=True, help="Exit on the first processing error.")
@click.option("--verbose", is_flag=True, help="Increase verbosity.")
@click.option("--log-all", is_flag=True, help="Enable logging for all modules.")
@click.argument("filenames", nargs=-1, required=True)
def run(cfg, output_dir, tle_dir, debug, verbose, log_all, filenames):
    """Read and calibrate AVHRR GAC data and write netCDF files."""
    logging_on(logging.DEBUG if verbose else logging.INFO, for_all=log_all)
    config = read_config(cfg)
    if output_dir:
        config["output"]["output_dir"] = output_dir
    if tle_dir:
        config["controls"]["reader_kwargs"]["tle_dir"] = tle_dir
    if debug:
        config["controls"]["debug"] = debug

    satpy.CHUNK_SIZE = config["controls"].get("pytroll_chunk_size", 1024)
    for filename in filenames:
        if tarfile.is_tarfile(filename):
            process_tarball(filename, config)
        else:
            process_file(filename, config)


def parse_date(ctx, param, value):
    if value is None:
        return None
    try:
        return isoparse(value).date()
    except (ValueError, TypeError) as exc:
        raise click.BadParameter(str(exc)) from exc


@cli.command()
@click.argument("filename")
@click.option(
    "--date", callback=parse_date, help="Only include observations on this date."
)
@click.option(
    "--where",
    type=click.Choice((CROP_OVERLAP_BEGINNING, CROP_OVERLAP_END)),
    default="end",
    help="Where to crop overlap.",
)
def crop_command(filename, date, where):
    """Crop a pygac-fdr output file to account for overlap."""
    start_line, end_line = crop(filename, where, date)
    click.echo(start_line, end_line)


@cli.command()
@click.argument("dbfile")
@click.option("-n", type=int, help="Maximum number of rows to print.")
def dump(dbfile, n):
    """Print the contents of a metadata database."""
    collector = MetadataCollector()
    mda = collector.read_sql(dbfile)
    pd.set_option("display.max_rows", n)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", 22)
    click.echo(mda)


@cli.group()
def mda():
    """Collect and update metadata."""


@mda.command("collect", cls=ArgumentsFileCommand)
@click.option("--dbfile", required=True, help="Metadata database to write.")
@click.option(
    "--if-exists",
    type=click.Choice(("append", "fail", "replace")),
    default="fail",
    help="What to do if the database table already exists.",
)
@click.option("--verbose", is_flag=True, help="Increase verbosity.")
@click.argument("filenames", nargs=-1, required=True)
def mda_collect(dbfile, if_exists, verbose, filenames):
    """Collect metadata from level 1c files. Accepts @filename lists."""
    logging_on(logging.DEBUG if verbose else logging.INFO)
    collector = MetadataCollector()
    mda = collector.collect_metadata(filenames)
    enhancer = MetadataEnhancer()
    mda = enhancer.enhance_metadata(mda)
    save_metadata_to_database(mda, dbfile, if_exists)


@mda.command("update")
@click.option(
    "--dbfile", required=True, help="Metadata database created by mda collect."
)
@click.option("--verbose", is_flag=True, help="Increase verbosity.")
def mda_update(dbfile, verbose):
    """Update metadata in level 1c files."""
    logging_on(logging.DEBUG if verbose else logging.INFO)
    metadata = read_metadata_from_database(dbfile)
    updater = MetadataUpdater()
    updater.update(metadata)


@cli.group()
def val():
    """Collect and plot validation statistics."""


@val.command("collect", cls=ArgumentsFileCommand)
@click.option("--dbfile", required=True, help="Statistics database to write.")
@click.option("--worker", type=int, default=1, help="Number of worker processes.")
@click.option("--verbose", is_flag=True, help="Increase verbosity.")
@click.argument("filenames", nargs=-1, required=True)
def val_collect(dbfile, worker, verbose, filenames):
    """Collect validation statistics. Accepts @filename lists."""
    logging_on(logging.DEBUG if verbose else logging.INFO)
    import sqlalchemy

    engine = sqlalchemy.create_engine(f"sqlite:///{dbfile}")
    collector = validation.StatsCollector(worker, engine)
    collector.get_statistics(filenames)


@val.command("plot")
@click.argument("dbfiles", nargs=-1, required=True)
@click.option("--cache", default="./cache", help="Cache directory.")
@click.option("--plots", default="./plots", help="Plots directory.")
@click.option("--verbose", is_flag=True, help="Increase verbosity.")
def val_plot(dbfiles, cache, plots, verbose):
    """Plot validation statistics from level 1c files."""
    global plots_dir
    logging_on(logging.DEBUG if verbose else logging.INFO)

    cache_dir = pathlib.Path(cache)
    cache_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = pathlib.Path(cache)
    plots_dir.mkdir(parents=True, exist_ok=True)

    validation.plots_dir = plots_dir
    validation.plot_zone_averages(dbfiles, cache_dir, plots_dir)
    validation.plot_hovmoeller(dbfiles, cache_dir)
