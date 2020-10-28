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

"""Fetch test data (input & reference output) from data server.

Existing files will only be re-downloaded if the server has a newer version.
"""

import logging
import os
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import yaml

from pygac_fdr.utils import LOGGER_NAME, logging_on

LOG = logging.getLogger(LOGGER_NAME)


def fetch_test_data(url, target_dir):
    LOG.info("Fetching test data")
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)
    url_p = urlparse(url)
    cut_dirs = len(url_p.path.strip("/").split("/"))
    cmd = [
        "wget",
        "--no-verbose",
        "--mirror",
        "--no-host-directories",
        "--no-parent",
        "--cut-dirs",
        str(cut_dirs),
        '--reject="index.html*"',
        url,
    ]
    subprocess.run(cmd, cwd=target_dir, check=True)


if __name__ == "__main__":
    logging_on(logging.INFO)
    with open(Path(__file__).parent / "test_end2end.yaml") as fh:
        config = yaml.safe_load(fh)
    fetch_test_data(config["test_data_url"], config["test_data_dir"])
