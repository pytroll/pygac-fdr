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

import os

from setuptools import find_packages, setup

try:
    # HACK: https://github.com/pypa/setuptools_scm/issues/190#issuecomment-351181286
    # Stop setuptools_scm from including all repository files
    import setuptools_scm.integration

    setuptools_scm.integration.find_files = lambda _: []
except ImportError:
    pass


if __name__ == "__main__":
    requires = [
        "setuptools_scm",
        "numpy",
        "xarray >=0.15.1",
        "pandas >=1.0.3",
        "netCDF4",
        "h5py",
        "pygac >=1.3.1",
        "satpy",
        "pyyaml",
        "trollsift",
        "fsspec",
    ]
    extras_require = {
        "tests": [
            "cfchecker >=4.1.0",
            "pytest",
            "pytest-cov",
            "pytest-testconfig",
            "matplotlib",
        ],
        "dev": ["pre-commit"],
    }
    README = open("README.md", "r").read()
    setup(
        name="pygac-fdr",
        description="Python package for creating a Fundamental Data Record (FDR) of AVHRR GAC "
        "data using pygac",
        long_description=README,
        long_description_content_type="text/markdown",
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: GNU General Public License v3 "
            + "or later (GPLv3+)",
            "Operating System :: OS Independent",
            "Programming Language :: Python",
            "Topic :: Scientific/Engineering",
        ],
        author="The Pytroll Team",
        author_email="pytroll@googlegroups.com",
        url="https://github.com/pytroll/pygac-fdr",
        packages=find_packages(),
        scripts=[os.path.join("bin", item) for item in os.listdir("bin")],
        install_requires=requires,
        extras_require=extras_require,
        python_requires=">=3.8",
    )
