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
"""Print contents of metadata database."""

import argparse

import pandas as pd

from pygac_fdr.metadata import MetadataCollector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dbfile", help="Metadata database")
    parser.add_argument("-n", help="Maximum number of rows to be printed", type=int)
    args = parser.parse_args()

    collector = MetadataCollector()
    mda = collector.read_sql(args.dbfile)

    pd.set_option("display.max_rows", args.n)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", 22)

    print(mda)
