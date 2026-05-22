"""Config file parsing."""

import yaml


def read_config(filename):
    with open(filename) as fh:
        config = yaml.safe_load(fh)

    # Add empty dictionaries for required sections
    for required in ["controls", "output", "netcdf"]:
        if required not in config or config[required] is None:
            config[required] = {}

    return config
