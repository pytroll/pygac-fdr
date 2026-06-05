"""Read and calibrate AVHRR GAC level 1b data."""

import os

import satpy
import trollsift

BANDS = ["1", "2", "3", "3a", "3b", "4", "5"]
AUX_DATA = [
    "latitude",
    "longitude",
    "qual_flags",
    "sensor_zenith_angle",
    "solar_zenith_angle",
    "solar_azimuth_angle",
    "sensor_azimuth_angle",
    "sun_sensor_azimuth_difference_angle",
]
GAC_FORMAT = (
    "{creation_site:3s}.{transfer_mode:4s}.{platform_id:2s}.D{start_time:%y%j.S%H%M}."
    "E{end_time:%H%M}.B{orbit_number:05d}{end_orbit_last_digits:02d}.{station:2s}"
)


def read_gac(filename, reader_kwargs=None):
    """Read and calibrate AVHRR GAC level 1b data using satpy.

    Args:
        filename (str): AVHRR GAC level 1b file
        reader_kwargs (dict): Keyword arguments to be passed to the reader.
    Returns:
        The loaded data in a satpy.Scene object.
    """
    scene = satpy.Scene(
        filenames=[filename], reader="avhrr_l1b_gaclac", reader_kwargs=reader_kwargs
    )
    scene.load(BANDS)
    scene.load(AUX_DATA)

    # Add additional metadata
    basename = os.path.basename(filename)
    fname_info = trollsift.parse(GAC_FORMAT, basename)
    orbit_number_end = (
        fname_info["orbit_number"] // 100 * 100 + fname_info["end_orbit_last_digits"]
    )
    scene.attrs.update(
        {
            "gac_filename": basename,
            "orbit_number_start": fname_info["orbit_number"],
            "orbit_number_end": orbit_number_end,
            "ground_station": fname_info["station"],
        }
    )

    return scene
