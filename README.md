# pygac-fdr
Python package for creating a Fundamental Data Record (FDR) of AVHRR GAC data using pygac

[![Build](https://travis-ci.com/pytroll/pygac-fdr.svg?branch=master)](https://travis-ci.com/github/pytroll/pygac-fdr?branch=master)
[![Coverage Status](https://coveralls.io/repos/github/pytroll/pygac-fdr/badge.svg?branch=master)](https://coveralls.io/github/pytroll/pygac-fdr?branch=master)
[![PyPI version](https://badge.fury.io/py/pygac-fdr.svg)](https://badge.fury.io/py/pygac-fdr)

Installation
============

To install the latest release:
```
pip install pygac-fdr
```

To install the latest development version:
```
pip install git+https://github.com/pytroll/pygac-fdr
```

Usage
=====

To read and calibrate AVHRR GAC level 1b data, adapt the config template in `etc/pygac-fdr.yaml`, then
run:
```
pygac-fdr-run --cfg=my_config.yaml /data/avhrr_gac/NSS.GHRR.M1.D20021.S0*
```

Results are written into the specified output directory in netCDF format. Afterwards, collect and
complement metadata of the generated netCDF files:

```
pygac-fdr-mda-collect --dbfile=test.sqlite3 /data/avhrr_gac/output/*
```

This might take some time, so the results are saved into a database. You can specify files from
multiple platforms; the metadata are analyzed for each platform separately. Finally, update the
netCDF metadata inplace:

```
pygac-fdr-mda-update --dbfile=test.sqlite3
```

Utilities for AVHRR GAC FDR Users
=================================

Cropping Overlap
----------------

Due to the data reception mechanism consecutive AVHRR GAC files often partly contain the same information. This is what
we call overlap. For example some scanlines in the end of file A also occur in the beginning of file B. The
`overlap_free_start` and `overlap_free_end` attributes in `pygac-fdr` output files indicate that overlap. There are two
ways to remove it:

- Cut overlap with subsequent file: Select scanlines `0:overlap_free_end`
- Cut overlap with preceding file: Select scanlines `overlap_free_start:-1`

If, in addition, users want to create daily composites, a file containing observations from two days has to be used
twice: Once only the part before UTC 00:00, and once only the part after UTC 00:00. Cropping overlap and day together
is a little bit more complex, because the overlap might cover UTC 00:00. That is why the `pygac-fdr-crop` utility is
provided:

```
$ pygac-fdr-crop AVHRR-GAC_FDR_1C_N06_19810330T225108Z_19810331T003506Z_...nc --date 19810330
0 8260
$ pygac-fdr-crop AVHRR-GAC_FDR_1C_N06_19810330T225108Z_19810331T003506Z_...nc --date 19810331
8261 12472
```

The returned numbers are start- and end-scanline (0-based).