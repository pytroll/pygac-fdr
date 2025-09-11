#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2020 pygac-fdr developers
#
# This file is part of pygac-fdr.
#
# pygac-fdr is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# pygac-fdr is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# pygac-fdr. If not, see <http://www.gnu.org/licenses/>.
"""Plot validation statistics from level 1c files"""

import argparse
import itertools as it
import logging
import pathlib

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
import sqlalchemy

from pygac_fdr.utils import LOGGER_NAME, logging_on


pd.plotting.register_matplotlib_converters()
LOG = logging.getLogger(LOGGER_NAME)


def plot_zone_averages(dbfiles, cache_dir, plots_dir):
    # collect data
    dfs = {}
    for dbfile in map(pathlib.Path, dbfiles):
        stem = dbfile.stem
        cache_file = cache_dir / f'{stem}_zone_averages.csv'
        if cache_file.is_file():
            df = pd.read_csv(cache_file, index_col=0, parse_dates=['timestamp'])
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
                parse_dates=['timestamp']
            )
            df.to_csv(cache_file)
        dfs[stem] = df
    # draw plots
    zones = ['day', 'night', 'twilight']
    channels = ['channel_1', 'channel_2', 'channel_3a', 'channel_3b', 'channel_4', 'channel_5']
    ylims = {
        'day': [(5, 25), (5, 25), (0, 20), (230, 325), (250, 300), (250, 300)],
        'night': [(0, 1), (0, 1), (0, 1), (260, 275), (255, 280), (255, 280)],
        'twilight': [(2, 10), (2, 10), (0, 10), (240, 340), (220, 340), (220, 320)]
    }
    idx = list(range(len(dfs)))
    color_idx = idx[::4] + idx[1::4] + idx[2::4] + idx[3::4]
    sorted_sats = pd.Series(
        {sat: df['timestamp'].quantile(0.5) for sat, df in dfs.items()}
    ).sort_values().index
    for zone, channel in it.product(zones, channels):
        ymin, ymax = ylims[zone][channels.index(channel)]
        fig, axarr = plt.subplots(nrows=3, figsize=(10,12), sharex=True, sharey=False)
        for i, sat in enumerate(sorted_sats):
            ax = axarr[0]
            df = dfs[sat]
            color_i = color_idx[i]
            color = plt.cm.jet(color_i/(len(dfs)-1))
            mask = df['zone'] == zone
            x = df.loc[mask, 'timestamp']
            y = df.loc[mask, channel]
            if y.empty or y.isna().all():
                continue
            sc = ax.scatter(x, y, s=2, color=color)
            x_text = x.quantile(0.5)
            scale = 0.05*(ymax - ymin)
            y_text = ymax - (3.5 - (i % 3))*scale
            ax.text(x_text, y_text, sat, color=color)
            ax.set_ylabel(f'{zone.capitalize()} Mean {channel.capitalize().replace("_"," ")}')
            ax.set_ylim(ymin, ymax)

            ax = axarr[1]
            if sat == 'N14':
                mask = mask & (df['timestamp'] > '1990')
            mask = mask & (df[channel] >= ymin) & (df[channel] <= ymax)
            if (~mask).all():
                continue
            data = df.loc[mask].set_index('timestamp').resample(
                pd.Timedelta(days=1)).interpolate()[channel]
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
            ax.set_ylabel(f'Fourier Smoothed ({i_ft} strongest freq)')
            ax.set_ylim(ymin, ymax)

            ax = axarr[2]
            df = dfs[sat]
            mask = df['zone'] == zone
            x = df.loc[mask, 'timestamp']
            y = df.loc[mask, channel + '_count']
            if y.empty or y.isna().all():
                continue
            sc = ax.scatter(x, y, s=2, color=color)
            x_text = x.quantile(0.5)
            y_text = 1 - 0.1*(3.5 - (i % 3))
            trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
            ax.text(x_text, y_text, sat, color=color, transform=trans)
            ax.set_ylabel(f'{zone.capitalize()} Counts {channel.capitalize().replace("_"," ")}')
        ax.set_xlabel('Date')
        [ax.grid() for ax in axarr]
        fig.autofmt_xdate()
        fig.patch.set_facecolor('white')
        plot_file = plots_dir / f"{zone}-{channel}.png"
        fig.savefig(plot_file, bbox_inches="tight")
        del fig


def plot_hovmoeller(dbfiles, cache_dir):
    # collect data
    dfs = {}
    for dbfile in map(pathlib.Path, dbfiles):
        stem = dbfile.stem
        cache_file = cache_dir / f'{stem}_hovmoeller.csv'
        if cache_file.is_file():
            df = pd.read_csv(cache_file, index_col=0, parse_dates=['timestamp'])
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
                parse_dates=['timestamp']
            )
            df.to_csv(cache_file)
        dfs[stem] = df
    # draw plots for morning and afternoon satellites
    channels = ['channel_1', 'channel_2', 'channel_3a', 'channel_3b', 'channel_4', 'channel_5']
    sats_a = ['TRN', 'N06', 'N08', 'N10', 'N12', 'N15', 'N17', 'M02', 'M01']
    sats_b = ['N07', 'N09', 'N11', 'N14', 'N16', 'N18', 'N19']
    for sats in [sats_a, sats_b]:
        all_df = None
        for sat in sats[1:]:
            if sat not in dfs:
                continue
            df = data[sat]
            df['satellite'] = sat
            if all_df is None:
                all_df = df
                continue
            all_df = all_df.loc[all_df['timestamp'] < df['timestamp'].min()].copy()
            all_df = all_df.append(df, ignore_index=True)
        all_df.reset_index(drop=True, inplace=True)
        for channel in channels:
            fig = draw_hovmoeller(channel, all_df)
            plot_file = plots_dir / f"plots/hovmoeller-{sats[0]}-{sats[-1]}-{channel}.png"
            fig.savefig(plot_file, bbox_inches="tight")


def draw_hovmoeller(channel, data):
    if channel.split('_')[1] in ('1', '2', '3a'):
        unit = '%'
    else:
        unit = 'K'
    fig, ax = plt.subplots(figsize=(16, 4))
    sc = ax.scatter(
        data['timestamp'],
        data['latitude'],
        c=data[channel],
        vmin=data[channel].quantile(0.90),
        vmax=data[channel].quantile(0.10),
        cmap=plt.get_cmap('coolwarm', 13)
    )
    ax.set_xlim(data['timestamp'].min(), data['timestamp'].max())
    ax.set_ylim(-90, 90)
    ax.set_xlabel('timestamp')
    ax.set_ylabel('latitude')
    sat_times = all_df.groupby('satellite').aggregate({'timestamp': ['min', 'max']})['timestamp']
    ax.vlines(sat_times['min'], -90, 90, linestyles='--', alpha=0.75)
    cbar = fig.colorbar(sc)
    cbar.set_label(f'{channel.replace("_", " ")} [{unit}]')
    text_pos = sat_times.apply(lambda row: row['min'] + (row['max']-row['min'])/2, axis=1)
    for sat, pos in text_pos.iteritems():
        ax.text(pos, 100, sat, horizontalalignment='center')
    fig.autofmt_xdate()
    fig.patch.set_facecolor('white')
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dbfiles", nargs="+", help="Statistics databases")
    parser.add_argument("--cache", help="Cache directory, defaults to %(default)s", default='./cache')
    parser.add_argument("--plots", help="Plots directory, defaults to %(default)s", default='./plots')
    parser.add_argument("--verbose", action="store_true", help="Increase verbosity")

    args = parser.parse_args()
    logging_on(logging.DEBUG if args.verbose else logging.INFO)

    # create cache and plots directory if not exists
    cache_dir = pathlib.Path(args.cache)
    cache_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = pathlib.Path(args.cache)
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_zone_averages(args.dbfiles, cache_dir, plots_dir)
    plot_hovmoeller(args.dbfiles, cache_dir, plots_dir)
