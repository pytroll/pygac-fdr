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
