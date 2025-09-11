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
"""Collect metadata from level 1c files"""

import argparse
import logging

from pygac_fdr.metadata import (
    MetadataCollector,
    MetadataEnhancer,
    save_metadata_to_database,
)
from pygac_fdr.utils import LOGGER_NAME, logging_on

LOG = logging.getLogger(LOGGER_NAME)

tooltip = """
tooltip: %(prog)s --dbfile example.db @example_filenames.txt;
for reading a long list of filenames from an arguments file.
"""

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, fromfile_prefix_chars="@", epilog=tooltip
    )
    parser.add_argument(
        "--dbfile", required=True, type=str, help="Metadata database to be written"
    )
    parser.add_argument(
        "--if-exists",
        choices=("append", "fail", "replace"),
        default="fail",
        help="What to do if database table already exists",
    )
    parser.add_argument("filenames", nargs="+", help="Level 1c files to be analyzed")
    parser.add_argument("--verbose", action="store_true", help="Increase verbosity")
    args = parser.parse_args()
    logging_on(logging.DEBUG if args.verbose else logging.INFO)

    collector = MetadataCollector()
    mda = collector.collect_metadata(args.filenames)
    enhancer = MetadataEnhancer()
    mda = enhancer.enhance_metadata(mda)
    save_metadata_to_database(mda, args.dbfile, args.if_exists)
