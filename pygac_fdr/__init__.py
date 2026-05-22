try:
    from pygac_fdr.version import version as __version__  # noqa
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "No module named pygac_fdr.version. This could mean "
        "you didn't install 'pygac_fdr' properly. Try reinstalling ('pip "
        "install')."
    )
try:
    # If the wheels of netCDF4 (used by this module) and h5py (imported by pygac) are incompatible,
    # segfaults or runtime errors like "NetCDF: HDF error" might occur. Prevent this by importing
    # netCDF4 first.
    import netCDF4  # noqa: F401
except ImportError:
    pass
