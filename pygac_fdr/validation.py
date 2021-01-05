import inspect
import logging
import multiprocessing

import netCDF4
import numpy as np
import pandas as pd


LOG = logging.getLogger(__package__)


class StatsCollector:
    """Collect statistics from level 1c files."""

    def __init__(self, processes, engine):
        """
        Args:
            processes: Number of worker processes
            engine: sqlalchemy engine
        """
        self.processes = processes
        self.engine = engine

    def get_statistics(self, filenames):
        """Collect statistics from the given level 1c files."""
        LOG.info("Collecting statistics")
        with multiprocessing.Pool(self.processes) as pool:
            results = pool.imap_unordered(self.collect_stats, filenames)
            for result in filter(None, results):
                self.store(result)

    @staticmethod
    def collect_stats(filename):
        """Collect statistics from a given level 1c file."""
        LOG.info("Collect statistics from %s", filename)
        with FileStatsExtractor(filename) as file_stats:
            result = file_stats.extract()
        return result

    def store(self, result):
        """Store the results in the given database."""
        with self.engine.connect() as con:
            for table, data in result.items():
                data.to_sql(table, con, if_exists='append', index=False)


class FileStatsExtractor:
    """Extract statistics from single level 1c file.
    Note: The statistics resulting from the extract method is a dictionary
        collecting the results of methods starting with 'extract_'.
        The result of such extract_<something> methods should be a DataFrame
        that will be added to the database, where <something> will also be the
        table name.
    """
    prefix = 'extract_'
    channels = [
        'reflectance_channel_1',
        'reflectance_channel_2',
        'reflectance_channel_3a',
        'brightness_temperature_channel_3b',
        'brightness_temperature_channel_4',
        'brightness_temperature_channel_5'
    ]

    def __init__(self, filename):
        """
        Args:
            filename: path to level 1c file.
        """
        self.filename = filename
        self._dataset = netCDF4.Dataset(filename)
        self._data = {}
        self._attributes = {}

    def __getitem__(self, key):
        if key not in self._data:
            self._data[key] = self._dataset[key][:].filled(np.nan)
        return self._data[key]

    def __enter__(self):
        if not self._dataset.isopen():
            self._dataset = netCDF4.Dataset(self.filename)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def get_attribute(self, key):
        """Get an attribute from level 1c file"""
        if key not in self._attributes:
            self._attributes[key] = self._dataset.getncattr(key)
        return self._attributes[key]

    def close(self):
        """Close level 1c file and clear caches."""
        self._dataset.close()
        self._data = {}
        self._attributes = {}

    def load_channels(self, flatten=False):
        """Load the channel data.
            Note: It will always return six channels. A single channel 3 will be
                called channel 3b and channel 3a will be filled with NaNs
        """
        channels = self.channels + ['brightness_temperature_channel_3']
        data = {}
        for channel in channels:
            try:
                data[channel] = self[channel]
            except (IndexError, KeyError):
                data[channel] = np.nan
            else:
                if flatten:
                    data[channel] = data[channel].flatten()
        # In case there is a channel 3, we call it channel 3b, to define six channels for all satellites.
        ch_3 = data.pop('brightness_temperature_channel_3')
        if not np.isnan(ch_3).all():
            data['brightness_temperature_channel_3b'] = ch_3
        return data

    def extract(self):
        """Extract all statistics."""
        result = {}
        for _name, method in inspect.getmembers(self.__class__, predicate=inspect.isfunction):
            if _name.startswith(self.prefix):
                name = _name[len(self.prefix):]
                LOG.debug("Extract %s from %s.", name, self.filename)
                try:
                    result[name] = method(self)
                except (RuntimeError, LookupError, ArithmeticError,
                        OSError, IOError, ValueError, TypeError):
                    LOG.exception('Could not extract %s from %s!', name, self.filename)
        return result

    def extract_hovmoeller(self):
        """Extract mean and count of values grouped by latitude bins of 1 degree."""
        data = pd.DataFrame(self.load_channels(flatten=True))
        data['latitude'] = self['latitude'].flatten()
        # latitude mean 1 degree steps
        data['lat_bin'] = 0.5 + np.floor(data['latitude'])
        groupby = data.groupby('lat_bin')
        hovmoeller = groupby[self.channels].agg(['mean', 'count'])
        hovmoeller.columns = ['_'.join(col).strip() for col in hovmoeller.columns.values]
        hovmoeller['timestamp'] = pd.Timestamp(self.get_attribute('start_time'))
        hovmoeller.reset_index(inplace=True)
        return hovmoeller

    def extract_zone_averages(self):
        """Extract day night twilight zone averages."""
        data = pd.DataFrame(self.load_channels(flatten=True))
        data['solar_zenith_angle'] = self['solar_zenith_angle'].flatten()
        # zones
        zones = pd.Series(['day', 'night', 'twilight'], name='zone')
        data['day'] = data['solar_zenith_angle'] < 90
        data['night'] = data['solar_zenith_angle'] >= 90
        data['twilight'] = ((data['solar_zenith_angle'] >= 80) & (data['solar_zenith_angle'] < 90))
        # add more interessting quantities
        data['diff_ch3bch5'] = data['brightness_temperature_channel_3b'] - data['brightness_temperature_channel_5']
        data['diff_ch4ch5'] = data['brightness_temperature_channel_4'] - data['brightness_temperature_channel_5']
        data['frac_ch2ch1'] = data['reflectance_channel_2'] / data['reflectance_channel_1']
        relevant = self.channels + ['diff_ch3bch5', 'diff_ch4ch5', 'frac_ch2ch1']
        means = pd.DataFrame(
            [data.loc[data[zone], relevant].mean() for zone in zones],
            index=zones
        )
        stds = pd.DataFrame(
            [data.loc[data[zone], relevant].std() for zone in zones],
            index=zones
        )
        counts = pd.DataFrame(
            [data.loc[data[zone], relevant].count() for zone in zones],
            index=zones
        )
        zone_averages = means.join(stds, lsuffix='_mean', rsuffix='').join(
            counts, lsuffix='_std', rsuffix='_count')
        zone_averages.reset_index(inplace=True)
        zone_averages['timestamp'] = pd.Timestamp(self.get_attribute('start_time'))
        return zone_averages
