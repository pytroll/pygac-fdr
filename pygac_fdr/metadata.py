from enum import IntEnum
from functools import lru_cache
import logging
import netCDF4
import numpy as np
import pandas as pd
import sqlite3
import xarray as xr


LOG = logging.getLogger(__name__)


class QualityFlags(IntEnum):
    OK = 0
    TOO_SHORT = 1  # too few scanlines
    TOO_LONG = 2  # (end_time - start_time) unrealistically large
    DUPLICATE = 3  # identical record from different ground stations
    REDUNDANT = 4  # subset of another file


FILL_VALUE_LINE = -9999


class MetadataCollector:
    """Collect and complement metadata from level 1c files.

    Additional metadata include global quality flags as well as cut information to remove
    orbit overlap.
    """

    def get_metadata(self, filenames):
        """Collect and complement metadata from the given level 1c files."""
        df = pd.DataFrame(self._collect_metadata(filenames))

        # Set quality flags
        df = df.groupby('platform').apply(lambda x: self._set_global_qual_flags(x))
        df = df.drop(['platform'], axis=1)

        # Calculate overlap
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
        mda['start_time'] = mda['start_time'].astype('datetime64[ns]')
        mda['end_time'] = mda['end_time'].astype('datetime64[ns]')
        return mda

    def _collect_metadata(self, filenames):
        """Collect metadata from the given level 1c files."""
        # TODO: Equator crossing time
        records = []
        for filename in filenames:
            with xr.open_dataset(filename) as ds:
                midnight_line = self._get_midnight_line(ds['acq_time'])
                rec = {'platform':  ds.attrs['platform_name'],
                       'start_time': ds['acq_time'].values[0],
                       'end_time': ds['acq_time'].values[-1],
                       'along_track': ds.dims['y'],
                       'filename': filename,
                       'midnight_line': midnight_line,
                       'cut_line_overlap': FILL_VALUE_LINE,
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
            return incr[0]
        return FILL_VALUE_LINE

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
        """Flag duplicate orbits in the given data frame.

        Two orbits are considered equal if platform, start- and end-time are identical. This happens
        if the same orbit has been transferred to two different ground stations.
        """
        gs_dupl = df.duplicated(subset=['platform', 'start_time', 'end_time'],
                                keep='first')
        df.loc[gs_dupl, 'global_quality_flag'] = QualityFlags.DUPLICATE

    def _set_too_short_flag(self, df, min_lines=50):
        """Flag short orbits in the given data frame.

        TODO: Is this necessary or do we want to keep these records?

        Args:
            min_lines (int): Minimum number of scanlines for an orbit to be considered ok. Otherwise
                             it will flagged as too short.
        """
        too_short = df['along_track'] < min_lines
        df.loc[too_short, 'global_quality_flag'] = QualityFlags.TOO_SHORT

    def _set_too_long_flag(self, df, max_length=120):
        """Flag orbits where (end_time - start_time) is unrealistically large.

        This happens if the timestamps of the first or last scanline are corrupted. Flag these
        cases to prevent that subsequent orbits are erroneously flagged as redundant.

        Args:
            max_length: Maximum length (minutes) for an orbit to be considered ok. Otherwise it
                        will be flagged as too long.
        """
        max_length = np.timedelta64(max_length, 'm')
        too_long = (df['end_time'] - df['start_time']) > max_length
        df.loc[too_long, 'global_quality_flag'] = QualityFlags.TOO_LONG

    def _set_global_qual_flags(self, df):
        """Set global quality flags."""
        df.sort_values(by=['start_time', 'end_time'], inplace=True)
        df = df.reset_index(drop=True)
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
            if this_row['end_time'] >= next_row['start_time']:
                # LRU cached method re-uses timestamps from the previous iteration.
                this_ds = self._read_acq_time(this_row['filename'])
                next_ds = self._read_acq_time(next_row['filename'])
                cut = (this_ds['acq_time'] >= next_ds['acq_time'][0]).argmax().values
                df.loc[df_ok.index[i], 'cut_line_overlap'] = cut
        return df


def update_metadata(mda):
    """Update metadata in level 1c files."""
    for _, row in mda.iterrows():
        with netCDF4.Dataset(filename=row['filename'], mode='r+') as nc:
            # Add global quality flag
            try:
                nc_qual_flag = nc.createVariable('global_quality_flag', datatype=np.uint8)
            except RuntimeError:
                nc_qual_flag = nc.variables['global_quality_flag']
            nc_qual_flag[:] = row['global_quality_flag']
            attrs = {'long_name': 'Global quality flag',
                     'comment': 'If this flag is everything else than "ok", it is recommended not '
                                'to use the file.',
                     'flag_values': [flag.value for flag in QualityFlags.__members__.values()],
                     'flag_meanings': [name.lower() for name in QualityFlags.__members__.keys()]}
            for key, val in attrs.items():
                nc_qual_flag.setncattr(key, val)

            # Add cut/midnight line
            special_lines = {
                'cut_line_overlap': 'Scanline (0-based) where to cut this orbit in order to remove '
                                    'overlap with the subsequent orbit',
                'midnight_line': 'Scanline (0-based) where UTC timestamp crosses the dateline'
            }
            for var_name, long_name in special_lines.items():
                try:
                    nc_var = nc.createVariable(var_name, datatype=np.int16,
                                               fill_value=FILL_VALUE_LINE)
                except RuntimeError:
                    nc_var = nc.variables[var_name]
                nc_var[:] = row[var_name]
                nc_var.setncattr('long_name', long_name)
