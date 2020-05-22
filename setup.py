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
import pygac_fdr


if __name__ == '__main__':
    requires = ['numpy', 'netCDF4', 'pygac >=1.3.1', 'satpy >=0.21.0', 'pyyaml']
    test_requires = ['pytest']
    README = open('README.md', 'r').read()
    setup(name='pygac-fdr',
          version=pygac_fdr.__version__,
          description='Python package for creating a Fundamental Data Record (FDR) of AVHRR GAC '
                      'data using pygac',
          long_description=README,
          author='The Pytroll Team',
          author_email='pytroll@googlegroups.com',
          url="https://github.com/pytroll/pygac-fdr",
          packages=find_packages(),
          scripts=[os.path.join('bin', item) for item in os.listdir('bin')],
          use_scm_version=True,
          install_requires=requires,
          tests_require=test_requires,
          python_requires='>=3.6',
          )
