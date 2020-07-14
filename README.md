# pygac-fdr
Python package for creating a Fundamental Data Record (FDR) of AVHRR GAC data using pygac

Installation
============

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
