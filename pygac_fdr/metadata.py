from enum import IntEnum
import numpy as np
import pandas as pd
import sqlite3
import xarray as xr


class QualityFlags(IntEnum):
    OK = 0
    TOO_SHORT = 1
    DUPLICATE = 2  # identical record from different ground stations
    REDUNDANT = 3  # subset of another file


records = [
    {'platform': 'NOAA-6',
     'start_time': np.datetime64('2009-06-30'),
     'end_time': np.datetime64('2009-07-01'),
     'along_track': 11999,
     'filename': 'foo.nc',
     'midnight_scanline': 1234,
     'cut_line_overlap': None,
     'quality_flag': QualityFlags.OK},
    {'platform': 'NOAA-6',
     'start_time': np.datetime64('2009-07-01'),
     'end_time': np.datetime64('2009-07-01 13:30'),
     'along_track': 12000,
     'filename': 'foo.nc',
     'midnight_scanline': 1234,
     'cut_line_overlap': None,
     'quality_flag': QualityFlags.OK},
    {'platform': 'NOAA-6',
     'start_time': np.datetime64('2009-07-01 12:00'),
     'end_time': np.datetime64('2009-07-01 13:00'),
     'along_track': 12001,
     'filename': 'foo.nc',
     'midnight_scanline': 1234,
     'cut_line_overlap': None,
     'quality_flag': QualityFlags.OK},
    {'platform': 'NOAA-6',
     'start_time': np.datetime64('2009-07-01 12:00'),
     'end_time': np.datetime64('2009-07-01 13:00'),
     'along_track': 12002,
     'filename': 'foo.nc',
     'midnight_scanline': 1234,
     'cut_line_overlap': None,
     'quality_flag': QualityFlags.OK},
    {'platform': 'NOAA-6',
     'start_time': np.datetime64('2009-07-02'),
     'end_time': np.datetime64('2009-07-03'),
     'along_track': 49,
     'filename': 'foo.nc',
     'midnight_scanline': 1234,
     'cut_line_overlap': None,
     'quality_flag': QualityFlags.OK},
    {'platform': 'NOAA-6',
     'start_time': np.datetime64('2009-07-03'),
     'end_time': np.datetime64('2009-07-04'),
     'along_track': 12003,
     'filename': 'foo.nc',
     'midnight_scanline': 1234,
     'cut_line_overlap': None,
     'quality_flag': QualityFlags.OK},

    {'platform': 'NOAA-7',
     'start_time': np.datetime64('2009-07-01'),
     'end_time': np.datetime64('2009-07-02'),
     'along_track': 9000,
     'filename': 'foo.nc',
     'midnight_scanline': 1234,
     'cut_line_overlap': None,
     'quality_flag': QualityFlags.OK},
    {'platform': 'NOAA-7',
     'start_time': np.datetime64('2009-07-01 12:00'),
     'end_time': np.datetime64('2009-07-01 13:00'),
     'along_track': 9001,
     'filename': 'foo.nc',
     'midnight_scanline': 1234,
     'cut_line_overlap': None,
     'quality_flag': QualityFlags.OK},
    {'platform': 'NOAA-7',
     'start_time': np.datetime64('2009-07-01 12:00'),
     'end_time': np.datetime64('2009-07-01 13:00'),
     'along_track': 9001,
     'filename': 'foo.nc',
     'midnight_scanline': 1234,
     'cut_line_overlap': None,
     'quality_flag': QualityFlags.OK},
    {'platform': 'NOAA-7',
     'start_time': np.datetime64('2009-07-02'),
     'end_time': np.datetime64('2009-07-03'),
     'along_track': 9002,
     'filename': 'foo.nc',
     'midnight_scanline': 1234,
     'cut_line_overlap': None,
     'quality_flag': QualityFlags.OK},
    {'platform': 'NOAA-7',
     'start_time': np.datetime64('2009-07-03'),
     'end_time': np.datetime64('2009-07-04'),
     'along_track': 49,
     'filename': 'foo.nc',
     'midnight_scanline': 1234,
     'cut_line_overlap': None,
     'quality_flag': QualityFlags.OK},
]


class PygacFdrMetadataCollector:
    def _collect_metadata(self, files):
        # TODO: Read actual file contents
        return records

    def _set_redundant_flag(self, df):
        def is_redundant(end_times):
            start_times = end_times.index.to_numpy()
            end_times = end_times.to_numpy()
            redundant = (start_times[-1] >= start_times) & \
                        (end_times[-1] <= end_times)
            redundant[-1] = False
            return redundant.any()

        # DataFrame.rolling only supports numerical data types. Convert timestamps to integer.
        df['start_time'] = df['start_time'].astype(np.int64)
        df['end_time'] = df['end_time'].astype(np.int64)

        # DataFrame.rolling().apply() only has access to one column at a time. Workaround: Move
        # start_time to the index and pass the end_time series - including the index - to our function.
        # This can be achieved by calling apply(..., raw=False).
        df_tmp = df.set_index('start_time')
        rolling = df_tmp['end_time'].rolling(20, min_periods=2)
        redundant = rolling.apply(is_redundant, raw=False).fillna(0).astype(np.bool)
        redundant.reset_index(drop=True, inplace=True)
        df.loc[redundant, 'quality_flag'] = QualityFlags.REDUNDANT

        df['start_time'] = df['start_time'].astype('datetime64[ns]')
        df['end_time'] = df['end_time'].astype('datetime64[ns]')

    def _set_duplicate_flag(self, df):
        gs_dupl = df.duplicated(subset=['platform', 'start_time', 'end_time'],
                                keep='first')
        df.loc[gs_dupl, 'quality_flag'] = QualityFlags.DUPLICATE

    def _set_too_short_flag(self, df, min_lines=50):
        # TODO: Do we want to keep these records?
        too_short = df['along_track'] < min_lines
        df.loc[too_short, 'quality_flag'] = QualityFlags.TOO_SHORT

    def set_qual_flags(self, df):
        df.sort_values(by=['start_time', 'end_time'], inplace=True)
        df.reset_index(inplace=True, drop=True)
        self._set_redundant_flag(df)
        self._set_duplicate_flag(df)
        self._set_too_short_flag(df)
        return df

    def calc_overlap(self, df):
        """
        Compare timestamps of two consecutive orbits and determine where they overlap. Cut the
        first one in the end so that there is no overlap. This reads the timestamps once from all
        files, but it's much simpler than the former logic using an approximate scanning
        frequency.
        """
        # TODO: Not tested
        df_qc = df[df['quality_flag'] == QualityFlags.OK]
        for i in range(1, len(df_qc)):
            # TODO: Re-use timestamps from previous iteration
            this_row = df_qc.iloc[i]
            prev_row = df_qc.iloc[i - 1]
            if this_row['start_time'] >= prev_row['end_time']:
                this_ds = xr.open_dataset(this_row['filename'])
                prev_ds = xr.open_dataset(prev_row['filename'])
                cut = (prev_ds['acq_time'] >= this_ds['acq_time'][0]).argmax()
                df[df_qc.index[i]]['cut_line_overlap'] = cut

    def get_metadata(self, files):
        df = pd.DataFrame(self._collect_metadata(files))

        # Set quality flags
        df = df.groupby('platform').apply(lambda x: self.set_qual_flags(x))
        df.drop(['platform'], axis=1, inplace=True)

        # Calculate overlap
        # TODO
        # self.calc_overlap(df)

        return df

    def to_sql(self, df, filename):
        con = sqlite3.connect(filename)
        df.to_sql(name='metadata', con=con, if_exists='replace')
        con.commit()
        con.close()


def set_metadata(files, mda):
    # TODO
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print(mda)


if __name__ == '__main__':
    # Collect metadata from nc files. Since this may take some time, dump results into a database.
    my_files = ['file1.nc', 'file2.nc']
    collector = PygacFdrMetadataCollector()
    mda = collector.get_metadata(my_files)
    collector.to_sql(mda, 'test.sqlite3')

    # Next step: Read database and set metadata of nc files. For smaller datasets you could pass the
    # metadata directly without dumping/loading it to/from a database.
    con = sqlite3.connect('test.sqlite3')
    mda = pd.read_sql('select * from metadata', con)
    set_metadata(files=my_files, mda=mda)
    con.close()
