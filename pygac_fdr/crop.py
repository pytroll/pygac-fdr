"""Crop file overlap."""

import datetime as dt

import xarray as xr
from dateutil.parser import isoparse

CROP_OVERLAP_BEGINNING = "beginning"
CROP_OVERLAP_END = "end"


def crop_end(ds, date=None):
    """Crop overlap with subsequent file.

    If requested, further crop the results by selecting observations of the given date only.
    This includes the following cases:

    a) Observations start and end at date=d. All scanlines can be used.
    b) Observations start at date=d and end at date=d+1. In this case only the part until
       date=d+1 00:00 is selected.
    c) Observations start at date=d-1 and end at date=d. Select the part from date=d 00:00 onwards,
       unless it lies within the overlap with the following file.

    Args:
        ds (xarray.Dataset): Dataset to be analyzed.
        date (datetime.date): Select observations of the given date only.

    Returns:
        Start scanline, end scanline (0-based)
    """
    overlap_free_end = int(ds["overlap_free_end"].values)  # 0-based

    # Cut overlap with subsequent file
    start_line = 0
    end_line = overlap_free_end

    if date:
        # Select observations belonging the given date
        day_before = date - dt.timedelta(days=1)
        start_date = isoparse(ds.attrs["start_time"]).date()
        try:
            midnight_line = int(ds["midnight_line"].values)  # 0-based
        except ValueError:
            midnight_line = None

        if start_date == date and not midnight_line:
            # a) All observations belong to the given date
            pass
        elif start_date == date and midnight_line:
            # b) Stop at date+1 00:00
            end_line = min(overlap_free_end, midnight_line)
        elif start_date == day_before and midnight_line:
            if overlap_free_end > midnight_line:
                # c.1) Start at date 00:00
                start_line = midnight_line + 1
            else:
                # c.2) All observations of the given date are within overlap
                start_line = end_line = None
        else:
            # No observations of the given date
            start_line = end_line = None

    return start_line, end_line


def crop_beginning(ds, date=None):
    # FUTURE
    raise NotImplementedError


def crop(filename, where, date=None):
    """Crop overlap.

    Args:
        filename: pygac-fdr output file to be analyzed.
        where: Specifies where to crop overlap. In the beginning (removes
               overlap with preceding file) or end (removes overlap with
               subsequent file).
        date: Select observations of the given date only.

    Returns:
        Start scanline, end scanline (0-based)
    """
    with xr.open_dataset(filename) as ds:
        if where == CROP_OVERLAP_END:
            return crop_end(ds, date)
        return crop_beginning(ds, date)
