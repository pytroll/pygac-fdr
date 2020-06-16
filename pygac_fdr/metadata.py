from enum import IntEnum
from functools import lru_cache
import logging
import netCDF4
import numpy as np
import pandas as pd
import sqlite3
import xarray as xr
from xarray.coding.times import encode_cf_datetime


LOG = logging.getLogger(__package__)


class QualityFlags(IntEnum):
    OK = 0
    INVALID_TIMESTAMP = 1  # end time < start time or timestamp out of valid range
    TOO_SHORT = 2  # not enough scanlines or duration too short
    TOO_LONG = 3  # (end_time - start_time) unrealistically large
    DUPLICATE = 4  # identical record from different ground stations
    REDUNDANT = 5  # subset of another file


FILL_VALUE_INT = -9999
FILL_VALUE_FLOAT = -9999.

ADDITIONAL_METADATA = [
    {'name': 'cut_line_overlap',
     'long_name': 'Scanline (0-based) where to cut this orbit in order to remove '
                  'overlap with the subsequent orbit',
     'dtype': np.int16,
     'fill_value': FILL_VALUE_INT},
    {'name': 'midnight_line',
     'long_name': 'Scanline (0-based) where UTC timestamp crosses the dateline',
     'dtype': np.int16,
     'fill_value': FILL_VALUE_INT},
    {'name': 'equator_crossing_longitude',
     'long_name': 'Longitude where ascending node crosses the equator',
     'units': 'degrees_east',
     'dtype': np.float64,
     'fill_value': FILL_VALUE_FLOAT},
    {'name': 'equator_crossing_time',
     'long_name': 'UTC time when ascending node crosses the equator',
     'units': 'seconds since 1970-01-01 00:00:00',
     'calendar': 'standard',
     'dtype': np.float64,
     'fill_value': FILL_VALUE_INT},
    {'name': 'global_quality_flag',
     'long_name': 'Global quality flag',
     'comment': 'If this flag is everything else than "ok", it is recommended not '
                'to use the file.',
     'flag_values': [flag.value for flag in QualityFlags.__members__.values()],
     'flag_meanings': [name.lower() for name in QualityFlags.__members__.keys()],
     'dtype': np.uint8,
     'fill_value': None}
]


class MetadataCollector:
    """Collect and complement metadata from level 1c files.

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
        self.min_duration = np.timedelta64(min_duration, 'm')

    def get_metadata(self, filenames):
        """Collect and complement metadata from the given level 1c files."""
        LOG.info('Collecting metadata')
        df = pd.DataFrame(self._collect_metadata(filenames))
        df.sort_values(by=['start_time', 'end_time'], inplace=True)

        # Set quality flags
        LOG.info('Computing quality flags')
        df = df.groupby('platform').apply(lambda x: self._set_global_qual_flags(x))
        df = df.drop(['platform'], axis=1)

        # Calculate overlap
        LOG.info('Computing overlap')
        df = df.groupby('platform').apply(lambda x: self._calc_overlap(x))

        return df

    def save_sql(self, mda, dbfile):
        """Save metadata to sqlite database."""
        con = sqlite3.connect(dbfile)
        mda.to_sql(name='metadata', con=con, if_exists='replace')
        con.commit()
        con.close()

    def read_sql(self, dbfile):
        """Read metadata from sqlite database."""
        with sqlite3.connect(dbfile) as con:
            mda = pd.read_sql('select * from metadata', con)
        mda = mda.set_index(['platform', 'level_1'])
        for time_col in ['start_time', 'end_time', 'equator_crossing_time']:
            mda[time_col] = mda[time_col].astype('datetime64[ns]')
        return mda

    def _collect_metadata(self, filenames):
        """Collect metadata from the given level 1c files."""
        records = []
        for filename in filenames:
            LOG.debug('Collecting metadata from {}'.format(filename))
            with xr.open_dataset(filename) as ds:
                midnight_line = self._get_midnight_line(ds['acq_time'])
                eq_cross_lon, eq_cross_time = self._get_equator_crossing(ds)
                rec = {'platform':  ds.attrs['platform'].split('>')[-1].strip(),
                       'start_time': ds['acq_time'].values[0],
                       'end_time': ds['acq_time'].values[-1],
                       'along_track': ds.dims['y'],
                       'filename': filename,
                       'equator_crossing_longitude': eq_cross_lon,
                       'equator_crossing_time': eq_cross_time,
                       'midnight_line': midnight_line,
                       'cut_line_overlap': np.nan,  # will be computed in a postprocessing
                       'global_quality_flag': QualityFlags.OK}
                records.append(rec)
        return records

    def _get_midnight_line(self, acq_time):
        """Find scanline where the UTC date increases by one day.

        Returns:
            int: The midnight scanline if it exists.
                 None, else.
        """
        d0 = np.datetime64('1970-01-01', 'D')
        days = (acq_time.astype('datetime64[D]') - d0).astype(np.int64)
        incr = np.where(np.diff(days) == 1)[0]
        if len(incr) >= 1:
            if len(incr) > 1:
                LOG.warning('UTC date increases more than once. Choosing the first '
                            'occurence as midnight scanline.')
            return float(incr[0])
        return np.nan

    def _get_equator_crossing(self, ds):
        """Determine where the ascending node crosses the equator.

        Returns:
            Longitude and UTC time
        """
        # Use coordinates in the middle of the swath
        mid_swath = ds['latitude'].shape[1] // 2
        lat = ds['latitude'].isel(x=mid_swath)
        lat_shift = lat.shift(y=-1, fill_value=lat.isel(y=-1))
        sign_change = np.sign(lat_shift) != np.sign(lat)
        ascending = lat_shift > lat
        lat_eq = lat.where(sign_change & ascending, drop=True)
        if len(lat_eq) > 0:
            return lat_eq['longitude'].values[0], lat_eq['acq_time'].values[0]
        return np.nan, np.datetime64('NaT')

    def _set_redundant_flag(self, df, window=20):
        """Flag redundant orbits in the given data frame.

        An orbit is called redundant if it is entirely overlapped by one of its predecessors
        (in time).

        Args:
            window (int): Number of preceding orbits to be taken into account
        """
        def is_redundant(end_times):
            start_times = end_times.index.get_level_values('start_time').to_numpy()
            end_times = end_times.to_numpy()
            redundant = (start_times[-1] >= start_times) & \
                        (end_times[-1] <= end_times)
            redundant[-1] = False
            return redundant.any()

        # Only take into account orbits that passed the QC check so far (e.g. we don't want
        # orbits flagged as TOO_LONG to overlap many subsequent orbits)
        df_ok = df[df['global_quality_flag'] == QualityFlags.OK]

        # DataFrame.rolling is an elegant solution, but it has two drawbacks:
        # a) It only supports numerical data types. Workaround: Convert timestamps to integer.
        df_ok['start_time'] = df_ok['start_time'].astype(np.int64)
        df_ok['end_time'] = df_ok['end_time'].astype(np.int64)

        # b) DataFrame.rolling().apply() only has access to one column at a time. Workaround: Move
        #    start_time to the index and pass the end_time series - including the index - to our
        #    function. This can be achieved by calling apply(..., raw=False).
        df_ok = df_ok.set_index('start_time', append=True)
        rolling = df_ok['end_time'].rolling(window, min_periods=2)
        redundant = rolling.apply(is_redundant, raw=False).fillna(0).astype(np.bool)
        redundant = redundant.reset_index('start_time', drop=True)

        # So far we have operated on the qc-passed rows only. Update quality flags of rows in the
        # original (full) data frame.
        redundant = redundant[redundant.astype(np.bool)]
        df.loc[redundant.index, 'global_quality_flag'] = QualityFlags.REDUNDANT

    def _set_duplicate_flag(self, df):
        """Flag duplicate files in the given data frame.

        Two files are considered equal if platform, start- and end-time are identical. This happens
        if the same measurement has been transferred to two different ground stations.
        """
        gs_dupl = df.duplicated(subset=['platform', 'start_time', 'end_time'],
                                keep='first')
        df.loc[gs_dupl, 'global_quality_flag'] = QualityFlags.DUPLICATE

    def _set_invalid_timestamp_flag(self, df):
        """Flag files with invalid timestamps.

        Timestamps are considered invalid if they are outside the valid range (1970-2030) or if
        end_time < start_time.
        """
        valid_min = np.datetime64('1970-01-01 00:00')
        valid_max = np.datetime64('2030-01-01 00:00')
        out_of_range = ((df['start_time'] < valid_min) |
                        (df['start_time'] > valid_max) |
                        (df['end_time'] < valid_min) |
                        (df['end_time'] > valid_max))
        neg_dur = df['end_time'] < df['start_time']
        invalid = neg_dur | out_of_range
        df.loc[invalid, 'global_quality_flag'] = QualityFlags.INVALID_TIMESTAMP

    def _set_too_short_flag(self, df):
        """Flag files considered too short.

        That means either not enough scanlines or duration is too short.
        """
        too_short = (
                (df['along_track'] < self.min_num_lines) |
                (abs(df['end_time'] - df['start_time']) < self.min_duration)
        )
        df.loc[too_short, 'global_quality_flag'] = QualityFlags.TOO_SHORT

    def _set_too_long_flag(self, df, max_length=120):
        """Flag files where (end_time - start_time) is unrealistically large.

        This happens if the timestamps of the first or last scanline are corrupted. Flag these
        cases to prevent that subsequent files are erroneously flagged as redundant.

        Args:
            max_length: Maximum length (minutes) for a file to be considered ok. Otherwise it
                        will be flagged as too long.
        """
        max_length = np.timedelta64(max_length, 'm')
        too_long = (df['end_time'] - df['start_time']) > max_length
        df.loc[too_long, 'global_quality_flag'] = QualityFlags.TOO_LONG

    def _set_global_qual_flags(self, df):
        """Set global quality flags."""
        df = df.reset_index(drop=True)
        self._set_invalid_timestamp_flag(df)
        self._set_too_short_flag(df)
        self._set_too_long_flag(df)
        self._set_redundant_flag(df)
        self._set_duplicate_flag(df)
        return df

    @lru_cache(maxsize=2)
    def _read_acq_time(self, filename):
        ds = xr.open_dataset(filename)
        return ds['acq_time'].compute()

    def _calc_overlap(self, df):
        """Compare timestamps of two consecutive orbits and determine where they overlap.

        Cut the first one in the end so that there is no overlap. This reads the timestamps once
        from all files.
        """
        df_ok = df[df['global_quality_flag'] == QualityFlags.OK]
        for i in range(0, len(df_ok)-1):
            this_row = df_ok.iloc[i]
            next_row = df_ok.iloc[i + 1]
            LOG.debug('Computing overlap for {}'.format(this_row['filename']))
            if this_row['end_time'] >= next_row['start_time']:
                # LRU cached method re-uses timestamps from the previous iteration.
                this_ds = self._read_acq_time(this_row['filename'])
                next_ds = self._read_acq_time(next_row['filename'])
                cut = (this_ds['acq_time'] >= next_ds['acq_time'][0]).argmax().values
                df.loc[df_ok.index[i], 'cut_line_overlap'] = cut
        return df


def update_metadata(mda):
    """Add additional metadata to level 1c files."""
    # Since xarray cannot modify files in-place, use netCDF4 directly. See
    # https://github.com/pydata/xarray/issues/2029.
    for _, row in mda.iterrows():
        LOG.debug('Updating metadata in {}'.format(row['filename']))
        with netCDF4.Dataset(filename=row['filename'], mode='r+') as nc:
            nc_acq_time = nc.variables['acq_time']
            for mda in ADDITIONAL_METADATA:
                mda = mda.copy()
                var_name = mda.pop('name')
                fill_value = mda.pop('fill_value')

                # Create nc variable
                try:
                    nc_var = nc.createVariable(var_name, datatype=mda.pop('dtype'),
                                               fill_value=fill_value)
                except RuntimeError:
                    # Variable already there
                    nc_var = nc.variables[var_name]

                # Write data to nc variable. Since netCDF4 cannot handle NaN nor NaT, disable
                # auto-masking, and set null-data to fill value manually. Furthermore, match
                # timestamp encoding with the acq_time variable.
                data = row[var_name]
                nc_var.set_auto_mask(False)
                if pd.isnull(data):
                    data = fill_value
                elif isinstance(data, pd.Timestamp):
                    data = encode_cf_datetime(data, units=nc_acq_time.units,
                                              calendar=nc_acq_time.calendar)[0]
                nc_var[:] = data

                # Set attributes of nc variable
                for key, val in mda.items():
                    nc_var.setncattr(key, val)
