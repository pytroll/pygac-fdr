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

"""L1b to l1c processor."""

from pygac_fdr.run.reader import read_gac
from pygac_fdr.run.writer import NetcdfWriter


def process_l1b_to_l1c(filename, config):
    """Process l1b file using pygac and save l1c to netcdf."""
    scene = read_gac(filename, reader_kwargs=config["controls"].get("reader_kwargs"))
    writer = NetcdfWriter(
        output_dir=config["output"].get("output_dir"),
        global_attrs=config.get("global_attrs"),
        gac_header_attrs=config.get("gac_header_attrs"),
        fname_fmt=config["output"].get("fname_fmt"),
        encoding=config["netcdf"].get("encoding"),
        engine=config["netcdf"].get("engine"),
        debug=config["controls"].get("debug"),
    )
    writer.write(scene=scene)
