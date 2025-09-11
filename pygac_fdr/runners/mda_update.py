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

import argparse
import logging

from pygac_fdr.metadata import MetadataUpdater, read_metadata_from_database
from pygac_fdr.utils import logging_on


def main():
    parser = argparse.ArgumentParser(description="Update metadata in level 1c files")
    parser.add_argument(
        "--dbfile",
        required=True,
        type=str,
        help="Metadata database created with pygac-fdr-mda-collect",
    )
    parser.add_argument("--verbose", action="store_true", help="Increase verbosity")
    args = parser.parse_args()
    logging_on(logging.DEBUG if args.verbose else logging.INFO)

    mda = read_metadata_from_database(args.dbfile)
    updater = MetadataUpdater()
    updater.update(mda)
