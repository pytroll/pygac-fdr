"""Collect and complement L1C metadata."""

import logging
import sqlite3
from datetime import datetime
from enum import IntEnum
from typing import Optional

import netCDF4
import numpy as np
import pandas as pd
import xarray as xr
from xarray.coding.times import encode_cf_datetime

LOG = logging.getLogger(__package__)


TIME_COVERAGE = {
    "METOP-A": (datetime(2007, 6, 28, 23, 14), None),
    "METOP-B": (datetime(2013, 1, 1, 1, 1), None),
    "METOP-C": (datetime(2018, 11, 7, 16, 28), None),
    "NOAA-6": (datetime(1980, 1, 1, 0, 0), datetime(1982, 8, 3, 0, 39)),
    "NOAA-7": (datetime(1981, 8, 24, 0, 13), datetime(1985, 2, 1, 22, 21)),
    "NOAA-8": (datetime(1983, 5, 4, 19, 9), datetime(1985, 10, 14, 3, 26)),
    "NOAA-9": (datetime(1985, 2, 25, 0, 13), datetime(1988, 11, 7, 21, 18)),
    "NOAA-10": (datetime(1986, 11, 17, 1, 22), datetime(1991, 9, 16, 21, 19)),
    "NOAA-11": (datetime(1988, 11, 8, 0, 16), datetime(1994, 10, 16, 23, 27)),
    "NOAA-12": (datetime(1991, 9, 16, 0, 17), datetime(1998, 12, 14, 20, 43)),
    "NOAA-14": (datetime(1995, 1, 20, 0, 37), datetime(2002, 10, 7, 22, 47)),
    "NOAA-15": (datetime(1998, 10, 26, 0, 54), None),
    "NOAA-16": (datetime(2001, 1, 1, 0, 0), datetime(2011, 12, 31, 23, 40)),
    "NOAA-17": (datetime(2002, 6, 25, 5, 41), datetime(2011, 12, 31, 19, 11)),
    "NOAA-18": (datetime(2005, 5, 20, 18, 17), None),
    "NOAA-19": (datetime(2009, 2, 6, 18, 32), None),
    "TIROS-N": (datetime(1978, 11, 5, 9, 8), datetime(1980, 1, 30, 17, 3)),
}  # Estimated based on NOAA L1B archive


class QualityFlags(IntEnum):
    OK = 0
    INVALID_TIMESTAMP = 1  # end time < start time or timestamp out of valid range
    TOO_SHORT = 2  # not enough scanlines or duration too short
    TOO_LONG = 3  # (end_time - start_time) unrealistically large
    DUPLICATE = 4  # identical record from different ground stations
    REDUNDANT = 5  # subset of another file


class MetadataCollector:
    """Collect metadata from level 1c files."""

    def collect_metadata(self, filenames):
        """Collect metadata from the given level 1c files."""
        records = []
        for filename in filenames:
            LOG.debug("Collecting metadata from {}".format(filename))
            with xr.open_dataset(filename) as ds:
                midnight_line = np.float64(self._get_midnight_line(ds["acq_time"]))
                eq_cross_lons, eq_cross_times = self._get_equator_crossings(ds)
                rec = {
                    "platform": ds.attrs["platform"].split(">")[-1].strip(),
                    "start_time": ds["acq_time"].values[0],
                    "end_time": ds["acq_time"].values[-1],
                    "along_track": ds.sizes["y"],
                    "filename": filename,
                    "orbit_number_start": ds.attrs["orbit_number_start"],
                    "orbit_number_end": ds.attrs["orbit_number_end"],
                    "equator_crossing_longitude_1": eq_cross_lons[0],
                    "equator_crossing_time_1": eq_cross_times[0],
                    "equator_crossing_longitude_2": eq_cross_lons[1],
                    "equator_crossing_time_2": eq_cross_times[1],
                    "midnight_line": midnight_line,
                    "overlap_free_start": np.nan,
                    "overlap_free_end": np.nan,
                    "global_quality_flag": QualityFlags.OK,
                }
                records.append(rec)
        return pd.DataFrame(records)

    def _get_midnight_line(self, acq_time):
        """Find scanline where the UTC date increases by one day.

        Returns:
            int: The midnight scanline if it exists.
                 None, else.
        """
        d0 = np.datetime64("1970-01-01", "D")
        days = (acq_time.astype("datetime64[D]") - d0) / np.timedelta64(1, "D")
        incr = np.where(np.diff(days) == 1)[0]
        if len(incr) >= 1:
            if len(incr) > 1:
                LOG.warning(
                    "UTC date increases more than once. Choosing the first "
                    "occurence as midnight scanline."
                )
            return incr[0]
        return np.nan

    def _get_equator_crossings(self, ds):
        """Determine where the ascending node(s) cross the equator.

        Returns:
            Longitudes and UTC times of first and second equator crossing, if any, NaN else.
        """
        lat = ds["latitude"].load()  # load dataset to prevent netCDF4 indexing error

        # Use coordinates in the middle of the swath
        mid_swath = lat.shape[1] // 2
        lat = lat.isel(x=mid_swath)
        lat_shift = lat.shift(y=-1, fill_value=lat.isel(y=-1))
        sign_change = np.sign(lat_shift) != np.sign(lat)
        ascending = lat_shift > lat
        lat_eq = lat.where(sign_change & ascending, drop=True)

        num_cross = min(2, len(lat_eq))  # Two crossings max
        eq_cross_lons = np.full(2, np.nan)
        eq_cross_times = np.full(
            2, np.datetime64("NaT", "ns"), dtype=lat_eq["acq_time"].dtype
        )
        eq_cross_lons[0:num_cross] = lat_eq["longitude"].values[0:num_cross]
        eq_cross_times[0:num_cross] = lat_eq["acq_time"].values[0:num_cross]
        return eq_cross_lons, eq_cross_times


class MetadataEnhancer:
    """Enhance metadata of level 1c files.

    Additional metadata include global quality flags as well equator crossing time and
    overlap information.
    """

    def __init__(self, min_num_lines=50, min_duration=5):
        """
        Args:
            min_num_lines: Minimum number of scanlines for a file to be considered ok. Otherwise
                           it will flagged as too short.
            min_duration: Minimum duration (in minutes) for a file to be considered ok. Otherwise
                          it will flagged as too short.
        """
        self.min_num_lines = min_num_lines
        self.min_duration = np.timedelta64(min_duration, "m")

    def enhance_metadata(self, df):
        """Complement metadata from level 1c files."""
        df = self._sort_by_ascending_time(df)
        df = self._set_global_quality_flag(df)
        df = self._calc_overlap(df)
        df = self._update_index(df)
        return df

    def _sort_by_ascending_time(self, df):
        return df.sort_values(by=["start_time", "end_time"])

    def _set_global_quality_flag(self, df):
        LOG.info("Computing quality flags")
        grouped = df.groupby("platform")
        return grouped.apply(
            lambda x: self._set_global_qual_flags_single_platform(x, x.name)
        )

    def _set_global_qual_flags_single_platform(self, df, platform):
        """Set global quality flags."""
        self._set_invalid_timestamp_flag(df, platform)
        self._set_too_short_flag(df)
        self._set_too_long_flag(df)
        self._set_duplicate_flag(df)
        self._set_redundant_flag(df)
        return df

    def _set_redundant_flag(self, df, window=20):
        """Flag redundant files in the given data frame.

        An file is called redundant if it is entirely overlapped by one of its predecessors
        (in time).

        Args:
            window (int): Number of preceding files to be taken into account

        TODO: Identify the following case as redundant, as it causes overlap_free_start to be
        greater than overlap_free_end:

        |-------|  previous file
            |--------|  current file
              |---------|  subsequent file
        """

        def is_redundant(end_times):
            start_times = end_times.index.get_level_values("start_time").to_numpy()
            end_times = end_times.to_numpy()
            this_start_time = start_times[-1]
            this_end_time = end_times[-1]
            redundant = (this_start_time >= start_times[:-1]) & (
                this_end_time <= end_times[:-1]
            )
            return redundant.any()

        # Only take into account files that passed the QC check so far (e.g. we don't want
        # files flagged as TOO_LONG to overlap many subsequent files)
        df_ok = df[df["global_quality_flag"] == QualityFlags.OK].copy()

        # Sort by ascending start time and descending end time. This is required to catch
        # redundant files with identical start times but different end times.
        df_ok = df_ok.sort_values(
            by=["start_time", "end_time"], ascending=[True, False]
        )

        # DataFrame.rolling is an elegant solution, but it has two drawbacks:
        # a) It only supports numerical data types. Workaround: Convert timestamps to integer.
        df_ok["start_time"] = df_ok["start_time"].astype(np.int64)
        df_ok["end_time"] = df_ok["end_time"].astype(np.int64)

        # b) DataFrame.rolling().apply() only has access to one column at a time. Workaround: Move
        #    start_time to the index and pass the end_time series - including the index - to our
        #    function. This can be achieved by calling apply(..., raw=False).
        df_ok = df_ok.set_index("start_time", append=True)
        rolling = df_ok["end_time"].rolling(window, min_periods=2)
        redundant = rolling.apply(is_redundant, raw=False).fillna(0).astype(bool)
        redundant = redundant.reset_index("start_time", drop=True)

        # So far we have operated on the qc-passed rows only. Update quality flags of rows in the
        # original (full) data frame.
        redundant = redundant[redundant.astype(bool)]
        df.loc[redundant.index, "global_quality_flag"] = QualityFlags.REDUNDANT

    def _set_duplicate_flag(self, df):
        """Flag duplicate files in the given data frame.

        Two files are considered equal if platform, start- and end-time are identical. This happens
        if the same measurement has been transferred to two different ground stations.
        """
        gs_dupl = df.duplicated(subset=["start_time", "end_time"], keep="first")
        df.loc[gs_dupl, "global_quality_flag"] = QualityFlags.DUPLICATE

    def _set_invalid_timestamp_flag(self, df, platform):
        """Flag files with invalid timestamps.

        Timestamps are considered invalid if they are outside the temporal coverage of the platform
        or if end_time < start_time.
        """
        valid_min, valid_max = TIME_COVERAGE[platform]
        if not valid_max:
            valid_max = np.datetime64("2030-01-01 00:00")
        valid_min = np.datetime64(valid_min)
        valid_max = np.datetime64(valid_max)
        out_of_range = (
            (df["start_time"] < valid_min)
            | (df["start_time"] > valid_max)
            | (df["end_time"] < valid_min)
            | (df["end_time"] > valid_max)
        )
        neg_dur = df["end_time"] < df["start_time"]
        invalid = neg_dur | out_of_range
        df.loc[invalid, "global_quality_flag"] = QualityFlags.INVALID_TIMESTAMP

    def _set_too_short_flag(self, df):
        """Flag files considered too short.

        That means either not enough scanlines or duration is too short.
        """
        too_short = (df["along_track"] < self.min_num_lines) | (
            abs(df["end_time"] - df["start_time"]) < self.min_duration
        )
        df.loc[too_short, "global_quality_flag"] = QualityFlags.TOO_SHORT

    def _set_too_long_flag(self, df, max_length=120):
        """Flag files where (end_time - start_time) is unrealistically large.

        This happens if the timestamps of the first or last scanline are corrupted. Flag these
        cases to prevent that subsequent files are erroneously flagged as redundant.

        Args:
            max_length: Maximum length (minutes) for a file to be considered ok. Otherwise it
                        will be flagged as too long.
        """
        max_length = np.timedelta64(max_length, "m")
        too_long = (df["end_time"] - df["start_time"]) > max_length
        df.loc[too_long, "global_quality_flag"] = QualityFlags.TOO_LONG

    def _calc_overlap(self, df):
        LOG.info("Computing overlap")
        grouped = df.groupby("platform", as_index=False, group_keys=False)
        return grouped.apply(self._calc_overlap_single_platform)

    def _calc_overlap_single_platform(self, df, open_end=False):
        """Compare timestamps of neighbouring files and determine overlap.

        For each file compare its timestamps with the start/end timestamps of the preceding and
        subsequent files. Determine the overlap-free part of the file and set the corresponding
        overlap_free_start/end attributes.
        """
        df_ok = df[df["global_quality_flag"] == QualityFlags.OK]

        for i in range(len(df_ok)):
            this_row = df_ok.iloc[i]
            prev_row = df_ok.iloc[i - 1] if i > 0 else None
            next_row = df_ok.iloc[i + 1] if i < len(df_ok) - 1 else None
            LOG.debug("Computing overlap for {}".format(this_row["filename"]))
            this_time = xr.open_dataset(this_row["filename"])["acq_time"]

            # Compute overlap with preceding file
            if prev_row is not None:
                if prev_row["end_time"] >= this_row["start_time"]:
                    prev_end_time = prev_row["end_time"].to_datetime64()
                    overlap_free_start = (
                        (this_time > prev_end_time).argmax(axis=0).values
                    )
                else:
                    overlap_free_start = 0
                df.loc[df_ok.index[i], "overlap_free_start"] = overlap_free_start
            else:
                # First file
                df.loc[df_ok.index[i], "overlap_free_start"] = 0

            # Compute overlap with subsequent file
            if next_row is not None:
                if this_row["end_time"] >= next_row["start_time"]:
                    next_start_time = next_row["start_time"].to_datetime64()
                    overlap_free_end = (this_time >= next_start_time).argmax(
                        axis=0
                    ).values - 1
                else:
                    overlap_free_end = this_row["along_track"] - 1
                df.loc[df_ok.index[i], "overlap_free_end"] = overlap_free_end
            elif not open_end:
                # Last file
                df.loc[df_ok.index[i], "overlap_free_end"] = this_row["along_track"] - 1

        return df

    def _update_index(self, df):
        df = self._make_multi_index(df)
        self._rename_multi_index(df)
        return df

    def _make_multi_index(self, df):
        if not isinstance(df.index, pd.MultiIndex):
            df = self._make_multi_index_single_platform(df)
        return df

    def _make_multi_index_single_platform(self, df):
        num_rec = len(df)
        platform_index = pd.Series([0] * num_rec)
        file_index = pd.Series(range(num_rec))
        return df.set_index([platform_index, file_index])

    def _rename_multi_index(self, df):
        df.index.names = ["platform_index", "file_index"]


def save_metadata_to_database(mda, dbfile, if_exists):
    """Save metadata to sqlite database."""
    con = sqlite3.connect(dbfile)
    mda.to_sql(name="metadata", con=con, if_exists=if_exists)
    con.commit()
    con.close()


def read_metadata_from_database(dbfile):
    """Read metadata from sqlite database."""
    with sqlite3.connect(dbfile) as con:
        mda = pd.read_sql("select * from metadata", con)
    mda = _restore_multi_index(mda)
    mda.fillna(value=np.nan, inplace=True)
    _cast_time_columns_to_datetime(mda)
    return mda


def _restore_multi_index(mda):
    return mda.set_index(["platform_index", "file_index"])


def _cast_time_columns_to_datetime(mda):
    for col in mda.columns:
        if "time" in col:
            mda[col] = mda[col].astype("datetime64[ns]")


class MetadataUpdater:
    scalar_vars = [
        "overlap_free_start",
        "overlap_free_end",
        "midnight_line",
        "global_quality_flag",
    ]
    array_vars = ["equator_crossing_longitude", "equator_crossing_time"]
    attrs = {
        "overlap_free_start": {
            "long_name": "First scanline (0-based) of the overlap-free part of this file. Scanlines "
            "before that also appear in the preceding file.",
        },
        "overlap_free_end": {
            "long_name": "Last scanline (0-based) of the overlap-free part of this file. Scanlines "
            "hereafter also appear in the subsequent file.",
        },
        "midnight_line": {
            "long_name": "Scanline (0-based) where UTC timestamp crosses the dateline",
        },
        "equator_crossing_longitude": {
            "long_name": "Longitude where ascending node crosses the equator",
            "comment": "Can happen twice per file",
            "units": "degrees_east",
        },
        "equator_crossing_time": {
            "long_name": "UTC time when ascending node crosses the equator",
            "comment": "Can happen twice per file",
            "units": "seconds since 1970-01-01 00:00:00",
            "calendar": "standard",
        },
        "global_quality_flag": {
            "long_name": "Global quality flag",
            "comment": 'If this flag is everything else than "ok", it is recommended not '
            "to use the file.",
            "flag_values": np.array(
                [flag.value for flag in QualityFlags.__members__.values()],
                dtype=np.uint8,
            ),
            "flag_meanings": " ".join(
                [name.lower() for name in QualityFlags.__members__.keys()]
            ),
        },
    }
    fill_value_int = -9999
    fill_value_float = -9999.0
    encoding = {
        "overlap_free_start": {"dtype": np.int16, "fill_value": fill_value_int},
        "overlap_free_end": {"dtype": np.int16, "fill_value": fill_value_int},
        "midnight_line": {"dtype": np.int16, "fill_value": fill_value_int},
        "equator_crossing_longitude": {
            "dtype": np.float64,
            "fill_value": fill_value_float,
        },
        "equator_crossing_time": {"dtype": np.float64, "fill_value": fill_value_float},
        "global_quality_flag": {"dtype": np.uint8, "fill_value": None},
    }

    def update(self, mda: pd.DataFrame) -> None:
        """Add additional metadata to level 1c files.

        Since xarray cannot modify files in-place, use netCDF4 directly. See
        https://github.com/pydata/xarray/issues/2029.

        Expected data frame layout:

                                               start_time                end_time  ...
        platform_index file_index                                                  ...
        NOAA-6         0          1981-03-30 00:54:21.206 1981-03-30 02:46:32.706  ...
                       1          1981-03-30 02:42:39.206 1981-03-30 04:27:59.706  ...
                       2          1981-03-30 04:23:58.206 1981-03-30 06:09:04.206  ...
        """
        for _, row in mda.iterrows():
            LOG.debug("Updating metadata in {}".format(row["filename"]))
            with netCDF4.Dataset(filename=row["filename"], mode="r+") as nc:
                self._update_file(nc=nc, row=row)

    def _update_file(self, nc: netCDF4.Dataset, row: pd.Series) -> None:
        """Update metadata of a single file.

        Expected columns in each row:

        start_time                                             1981-03-30 22:51:08.706000
        end_time                                               1981-03-31 00:35:07.206000
        along_track                                                                 12474
        filename                        test_data/output/normal/AVHRR-GAC_FDR_1C_N06_1...
        orbit_number_start                                                           9133
        orbit_number_end                                                             9134
        equator_crossing_longitude_1                                              -48.094
        equator_crossing_time_1                                1981-03-30 22:51:44.206000
        equator_crossing_longitude_2                                              -73.398
        equator_crossing_time_2                                1981-03-31 00:32:57.706000
        midnight_line                                                              8260.0
        overlap_free_start                                                          608.0
        overlap_free_end                                                          12473.0
        global_quality_flag                                                             0
        """
        self._add_scalar_vars(nc, row)
        self._add_array_vars(nc, row)

    def _add_scalar_vars(self, nc: netCDF4.Dataset, row: pd.Series) -> None:
        for var_name in self.scalar_vars:
            self._add_scalar_variable(nc, row[var_name], var_name)

    def _stack_eq_crossings(self, row: pd.Series) -> pd.DataFrame:
        """Stack equator crossings to simplify the netCDF file.

        Returns a data frame with this layout:

            equator_crossing_time  equator_crossing_longitude
        0 1981-03-30 00:55:43.706                     -79.094
        1 1981-03-30 02:36:57.706                    -104.398
        """
        return pd.DataFrame(
            {
                "equator_crossing_time": [
                    row["equator_crossing_time_1"],
                    row["equator_crossing_time_2"],
                ],
                "equator_crossing_longitude": [
                    row["equator_crossing_longitude_1"],
                    row["equator_crossing_longitude_2"],
                ],
            },
        )

    def _add_scalar_variable(
        self, nc: netCDF4.Dataset, data: np.ndarray, var_name: str
    ) -> None:
        nc_var = self._create_nc_var(
            nc=nc,
            var_name=var_name,
        )
        self._write_data(nc_var, self._encode_scalar(data, var_name))
        self._set_var_attrs(nc_var)

    def _create_nc_var(
        self,
        nc: netCDF4.Dataset,
        var_name: str,
        new_dims: Optional[dict[str, int]] = None,
    ) -> netCDF4.Variable:
        """Create netCDF variable and dimension."""
        if new_dims:
            for dim_name, size in new_dims.items():
                if dim_name not in nc.dimensions:
                    nc.createDimension(dim_name, size=size)
        if var_name in nc.variables:
            return nc.variables[var_name]
        return nc.createVariable(
            var_name,
            datatype=self.encoding[var_name]["dtype"],
            fill_value=self.encoding[var_name]["fill_value"],
            dimensions=tuple(new_dims.keys()) if new_dims else (),
        )

    def _encode_scalar(self, data: float | int | pd.Timestamp, var_name: str) -> float:
        fill_value = self.encoding[var_name]["fill_value"]
        attrs = self.attrs[var_name]
        if isinstance(data, pd.Timestamp):
            data, _, _ = encode_cf_datetime(
                data.to_datetime64(), units=attrs["units"], calendar=attrs["calendar"]
            )
        return np.nan_to_num(data, nan=fill_value)

    def _write_data(self, nc_var: netCDF4.Variable, encoded_data: np.ndarray) -> None:
        # Write data to nc variable. Since netCDF4 cannot handle NaN nor NaT, disable
        # auto-masking, and write encoded data directly.
        nc_var.set_auto_mask(False)
        nc_var[:] = encoded_data

    def _add_array_vars(self, nc: netCDF4.Dataset, row: pd.Series) -> None:
        crossings = self._stack_eq_crossings(row)
        for var_name in self.array_vars:
            self._add_array_var(
                nc=nc,
                data=crossings[var_name].values,
                var_name=var_name,
                new_dims={"num_eq_cross": 2},
            )

    def _add_array_var(
        self,
        nc: netCDF4.Dataset,
        data: np.ndarray,
        var_name: str,
        new_dims: dict[str, int],
    ) -> None:
        nc_var = self._create_nc_var(nc=nc, var_name=var_name, new_dims=new_dims)
        nc_var.set_auto_mask(False)
        nc_var[:] = self._encode_array(data, var_name)
        self._set_var_attrs(nc_var)

    def _encode_array(self, data: np.ndarray, var_name: str) -> np.ndarray:
        fill_value = self.encoding[var_name]["fill_value"]
        attrs = self.attrs[var_name]
        if np.issubdtype(data.dtype, np.datetime64):
            if np.all(np.isnan(data)):
                # xarray fails to encode all-NaT array
                encoded = np.full(data.shape, np.nan)
            else:
                encoded, _, _ = encode_cf_datetime(
                    data,
                    units=attrs["units"],
                    calendar=attrs["calendar"],
                    dtype=self.encoding[var_name]["dtype"],
                )
        else:
            encoded = data
        return np.nan_to_num(encoded, nan=fill_value)

    def _set_var_attrs(self, nc_var: netCDF4.Variable) -> None:
        attrs = self.attrs[nc_var.name]
        for key, val in attrs.items():
            nc_var.setncattr(key, val)
