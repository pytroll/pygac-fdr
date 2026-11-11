import inspect
import itertools as it
import logging
import multiprocessing
import pathlib

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
                data.to_sql(table, con, if_exists="append", index=False)


class FileStatsExtractor:
    """Extract statistics from single level 1c file.
    Note: The statistics resulting from the extract method is a dictionary
        collecting the results of methods starting with 'extract_'.
        The result of such extract_<something> methods should be a DataFrame
        that will be added to the database, where <something> will also be the
        table name.
    """

    prefix = "extract_"
    channels = [
        "reflectance_channel_1",
        "reflectance_channel_2",
        "reflectance_channel_3a",
        "brightness_temperature_channel_3b",
        "brightness_temperature_channel_4",
        "brightness_temperature_channel_5",
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
        channels = self.channels + ["brightness_temperature_channel_3"]
        data = {}
        for channel in channels:
            try:
                data[channel] = self[channel]
            except IndexError, KeyError:
                data[channel] = np.nan
            else:
                if flatten:
                    data[channel] = data[channel].flatten()
        # In case there is a channel 3, we call it channel 3b, to define six channels for all satellites.
        ch_3 = data.pop("brightness_temperature_channel_3")
        if not np.isnan(ch_3).all():
            data["brightness_temperature_channel_3b"] = ch_3
        return data

    def extract(self):
        """Extract all statistics."""
        result = {}
        for _name, method in inspect.getmembers(
            self.__class__, predicate=inspect.isfunction
        ):
            if _name.startswith(self.prefix):
                name = _name[len(self.prefix) :]
                LOG.debug("Extract %s from %s.", name, self.filename)
                try:
                    result[name] = method(self)
                except (
                    RuntimeError,
                    LookupError,
                    ArithmeticError,
                    OSError,
                    IOError,
                    ValueError,
                    TypeError,
                ):
                    LOG.exception("Could not extract %s from %s!", name, self.filename)
        return result

    def extract_hovmoeller(self):
        """Extract mean and count of values grouped by latitude bins of 1 degree."""
        data = pd.DataFrame(self.load_channels(flatten=True))
        data["latitude"] = self["latitude"].flatten()
        # latitude mean 1 degree steps
        data["lat_bin"] = 0.5 + np.floor(data["latitude"])
        groupby = data.groupby("lat_bin")
        hovmoeller = groupby[self.channels].agg(["mean", "count"])
        hovmoeller.columns = [
            "_".join(col).strip() for col in hovmoeller.columns.values
        ]
        hovmoeller["timestamp"] = pd.Timestamp(self.get_attribute("start_time"))
        hovmoeller.reset_index(inplace=True)
        return hovmoeller

    def extract_zone_averages(self):
        """Extract day night twilight zone averages."""
        data = pd.DataFrame(self.load_channels(flatten=True))
        data["solar_zenith_angle"] = self["solar_zenith_angle"].flatten()
        # zones
        zones = pd.Series(["day", "night", "twilight"], name="zone")
        data["day"] = data["solar_zenith_angle"] < 90
        data["night"] = data["solar_zenith_angle"] >= 90
        data["twilight"] = (data["solar_zenith_angle"] >= 80) & (
            data["solar_zenith_angle"] < 90
        )
        # add more interessting quantities
        data["diff_ch3bch5"] = (
            data["brightness_temperature_channel_3b"]
            - data["brightness_temperature_channel_5"]
        )
        data["diff_ch4ch5"] = (
            data["brightness_temperature_channel_4"]
            - data["brightness_temperature_channel_5"]
        )
        data["frac_ch2ch1"] = (
            data["reflectance_channel_2"] / data["reflectance_channel_1"]
        )
        relevant = self.channels + ["diff_ch3bch5", "diff_ch4ch5", "frac_ch2ch1"]
        means = pd.DataFrame(
            [data.loc[data[zone], relevant].mean() for zone in zones], index=zones
        )
        stds = pd.DataFrame(
            [data.loc[data[zone], relevant].std() for zone in zones], index=zones
        )
        counts = pd.DataFrame(
            [data.loc[data[zone], relevant].count() for zone in zones], index=zones
        )
        zone_averages = means.join(stds, lsuffix="_mean", rsuffix="").join(
            counts, lsuffix="_std", rsuffix="_count"
        )
        zone_averages.reset_index(inplace=True)
        zone_averages["timestamp"] = pd.Timestamp(self.get_attribute("start_time"))
        return zone_averages


def plot_zone_averages(dbfiles, cache_dir, plots_dir):
    import matplotlib.pyplot as plt
    import matplotlib.transforms as transforms
    import sqlalchemy

    pd.plotting.register_matplotlib_converters()
    dfs = {}
    for dbfile in map(pathlib.Path, dbfiles):
        stem = dbfile.stem
        cache_file = cache_dir / f"{stem}_zone_averages.csv"
        if cache_file.is_file():
            df = pd.read_csv(cache_file, index_col=0, parse_dates=["timestamp"])
        else:
            engine = sqlalchemy.create_engine(dbfile)
            df = pd.read_sql(
                """
                SELECT
                    zone,
                    DATE(timestamp) AS 'timestamp',
                    SUM(reflectance_channel_1_mean * reflectance_channel_1_count)
                        / SUM(reflectance_channel_1_count) AS 'channel_1',
                    SUM(reflectance_channel_1_count) AS 'channel_1_count',
                    SUM(reflectance_channel_2_mean * reflectance_channel_2_count)
                        / SUM(reflectance_channel_2_count) AS 'channel_2',
                    SUM(reflectance_channel_2_count) AS 'channel_2_count',
                    SUM(reflectance_channel_3a_mean * reflectance_channel_3a_count)
                        / SUM(reflectance_channel_3a_count) AS 'channel_3a',
                    SUM(reflectance_channel_3a_count) AS 'channel_3a_count',
                    SUM(brightness_temperature_channel_3b_mean * brightness_temperature_channel_3b_count)
                        / SUM(brightness_temperature_channel_3b_count) AS 'channel_3b',
                    SUM(brightness_temperature_channel_3b_count) AS 'channel_3b_count',
                    SUM(brightness_temperature_channel_4_mean * brightness_temperature_channel_4_count)
                        / SUM(brightness_temperature_channel_4_count) AS 'channel_4',
                    SUM(brightness_temperature_channel_4_count) AS 'channel_4_count',
                    SUM(brightness_temperature_channel_5_mean * brightness_temperature_channel_5_count)
                        / SUM(brightness_temperature_channel_5_count) AS 'channel_5',
                    SUM(brightness_temperature_channel_5_count) AS 'channel_5_count',
                    SUM(diff_ch3bch5_mean * diff_ch3bch5_count) / SUM(diff_ch3bch5_count) AS 'diff_ch3bch5',
                    SUM(diff_ch3bch5_count) AS 'diff_ch3bch5_count',
                    SUM(diff_ch4ch5_mean * diff_ch4ch5_count) / SUM(diff_ch4ch5_count) AS 'diff_ch4ch5',
                    SUM(diff_ch4ch5_count) AS 'diff_ch4ch5_count',
                    SUM(frac_ch2ch1_mean * frac_ch2ch1_count) / SUM(frac_ch2ch1_count) AS 'frac_ch2ch1',
                    SUM(frac_ch2ch1_count) AS 'frac_ch2ch1_count'
                FROM
                    zone_averages
                GROUP BY
                    zone, DATE(timestamp)
                ORDER BY
                    timestamp
                """,
                engine,
                parse_dates=["timestamp"],
            )
            df.to_csv(cache_file)
        dfs[stem] = df

    zones = ["day", "night", "twilight"]
    channels = [
        "channel_1",
        "channel_2",
        "channel_3a",
        "channel_3b",
        "channel_4",
        "channel_5",
    ]
    ylims = {
        "day": [(5, 25), (5, 25), (0, 20), (230, 325), (250, 300), (250, 300)],
        "night": [(0, 1), (0, 1), (0, 1), (260, 275), (255, 280), (255, 280)],
        "twilight": [(2, 10), (2, 10), (0, 10), (240, 340), (220, 340), (220, 320)],
    }
    idx = list(range(len(dfs)))
    color_idx = idx[::4] + idx[1::4] + idx[2::4] + idx[3::4]
    sorted_sats = (
        pd.Series({sat: df["timestamp"].quantile(0.5) for sat, df in dfs.items()})
        .sort_values()
        .index
    )
    for zone, channel in it.product(zones, channels):
        ymin, ymax = ylims[zone][channels.index(channel)]
        fig, axarr = plt.subplots(nrows=3, figsize=(10, 12), sharex=True, sharey=False)
        for i, sat in enumerate(sorted_sats):
            ax = axarr[0]
            df = dfs[sat]
            color_i = color_idx[i]
            color = plt.cm.jet(color_i / (len(dfs) - 1))
            mask = df["zone"] == zone
            x = df.loc[mask, "timestamp"]
            y = df.loc[mask, channel]
            if y.empty or y.isna().all():
                continue
            sc = ax.scatter(x, y, s=2, color=color)
            x_text = x.quantile(0.5)
            scale = 0.05 * (ymax - ymin)
            y_text = ymax - (3.5 - (i % 3)) * scale
            ax.text(x_text, y_text, sat, color=color)
            ax.set_ylabel(
                f'{zone.capitalize()} Mean {channel.capitalize().replace("_"," ")}'
            )
            ax.set_ylim(ymin, ymax)

            ax = axarr[1]
            if sat == "N14":
                mask = mask & (df["timestamp"] > "1990")
            mask = mask & (df[channel] >= ymin) & (df[channel] <= ymax)
            if (~mask).all():
                continue
            data = (
                df.loc[mask]
                .set_index("timestamp")
                .resample(pd.Timedelta(days=1))
                .interpolate()[channel]
            )
            x = data.index
            y = data
            ft = np.fft.rfft(y)
            ind = np.argsort(np.abs(ft))
            ft_0 = ft[0]
            i_ft = min(50, len(ft))
            ft[np.abs(ft) < np.abs(ft[ind[-i_ft]])] = 0
            ft[0] = ft_0
            smooth = np.fft.irfft(ft, len(y))
            ax.plot(x, smooth, color=color)
            ax.text(x_text, y_text, sat, color=color)
            ax.set_ylabel(f"Fourier Smoothed ({i_ft} strongest freq)")
            ax.set_ylim(ymin, ymax)

            ax = axarr[2]
            df = dfs[sat]
            mask = df["zone"] == zone
            x = df.loc[mask, "timestamp"]
            y = df.loc[mask, channel + "_count"]
            if y.empty or y.isna().all():
                continue
            sc = ax.scatter(x, y, s=2, color=color)
            x_text = x.quantile(0.5)
            y_text = 1 - 0.1 * (3.5 - (i % 3))
            trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
            ax.text(x_text, y_text, sat, color=color, transform=trans)
            ax.set_ylabel(
                f'{zone.capitalize()} Counts {channel.capitalize().replace("_"," ")}'
            )
        ax.set_xlabel("Date")
        [ax.grid() for ax in axarr]
        fig.autofmt_xdate()
        fig.patch.set_facecolor("white")
        plot_file = plots_dir / f"{zone}-{channel}.png"
        fig.savefig(plot_file, bbox_inches="tight")
        del fig


def plot_hovmoeller(dbfiles, cache_dir):
    import sqlalchemy

    pd.plotting.register_matplotlib_converters()
    dfs = {}
    for dbfile in map(pathlib.Path, dbfiles):
        stem = dbfile.stem
        cache_file = cache_dir / f"{stem}_hovmoeller.csv"
        if cache_file.is_file():
            df = pd.read_csv(cache_file, index_col=0, parse_dates=["timestamp"])
        else:
            engine = sqlalchemy.create_engine(dbfile)
            df = pd.read_sql(
                """
                SELECT
                    DATE(timestamp) AS 'timestamp',
                    lat_bin AS 'latitude',
                    SUM(reflectance_channel_1_mean * reflectance_channel_1_count)
                        / SUM(reflectance_channel_1_count) AS 'channel_1',
                    SUM(reflectance_channel_2_mean * reflectance_channel_2_count)
                        / SUM(reflectance_channel_2_count) AS 'channel_2',
                    SUM(reflectance_channel_3a_mean * reflectance_channel_3a_count)
                        / SUM(reflectance_channel_3a_count) AS 'channel_3a',
                    SUM(brightness_temperature_channel_3b_mean * brightness_temperature_channel_3b_count)
                        / SUM(brightness_temperature_channel_3b_count) AS 'channel_3b',
                    SUM(brightness_temperature_channel_4_mean * brightness_temperature_channel_4_count)
                        / SUM(brightness_temperature_channel_4_count) AS 'channel_4',
                    SUM(brightness_temperature_channel_5_mean * brightness_temperature_channel_5_count)
                        / SUM(brightness_temperature_channel_5_count) AS 'channel_5'
                FROM
                    hovmoeller
                GROUP BY
                    DATE(timestamp), lat_bin
                ORDER BY
                    timestamp
                """,
                engine,
                parse_dates=["timestamp"],
            )
            df.to_csv(cache_file)
        dfs[stem] = df

    channels = [
        "channel_1",
        "channel_2",
        "channel_3a",
        "channel_3b",
        "channel_4",
        "channel_5",
    ]
    sats_a = ["TRN", "N06", "N08", "N10", "N12", "N15", "N17", "M02", "M01"]
    sats_b = ["N07", "N09", "N11", "N14", "N16", "N18", "N19"]
    for sats in [sats_a, sats_b]:
        all_df = None
        for sat in sats[1:]:
            if sat not in dfs:
                continue
            df = data[sat]
            df["satellite"] = sat
            if all_df is None:
                all_df = df
                continue
            all_df = all_df.loc[all_df["timestamp"] < df["timestamp"].min()].copy()
            all_df = all_df.append(df, ignore_index=True)
        all_df.reset_index(drop=True, inplace=True)
        for channel in channels:
            fig = draw_hovmoeller(channel, all_df)
            plot_file = (
                plots_dir / f"plots/hovmoeller-{sats[0]}-{sats[-1]}-{channel}.png"
            )
            fig.savefig(plot_file, bbox_inches="tight")


def draw_hovmoeller(channel, data):
    import matplotlib.pyplot as plt

    pd.plotting.register_matplotlib_converters()
    if channel.split("_")[1] in ("1", "2", "3a"):
        unit = "%"
    else:
        unit = "K"
    fig, ax = plt.subplots(figsize=(16, 4))
    sc = ax.scatter(
        data["timestamp"],
        data["latitude"],
        c=data[channel],
        vmin=data[channel].quantile(0.90),
        vmax=data[channel].quantile(0.10),
        cmap=plt.get_cmap("coolwarm", 13),
    )
    ax.set_xlim(data["timestamp"].min(), data["timestamp"].max())
    ax.set_ylim(-90, 90)
    ax.set_xlabel("timestamp")
    ax.set_ylabel("latitude")
    sat_times = all_df.groupby("satellite").aggregate({"timestamp": ["min", "max"]})[
        "timestamp"
    ]
    ax.vlines(sat_times["min"], -90, 90, linestyles="--", alpha=0.75)
    cbar = fig.colorbar(sc)
    cbar.set_label(f'{channel.replace("_", " ")} [{unit}]')
    text_pos = sat_times.apply(
        lambda row: row["min"] + (row["max"] - row["min"]) / 2, axis=1
    )
    for sat, pos in text_pos.iteritems():
        ax.text(pos, 100, sat, horizontalalignment="center")
    fig.autofmt_xdate()
    fig.patch.set_facecolor("white")
    return fig
