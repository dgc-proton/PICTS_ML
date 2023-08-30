"""
This module contains shared functions used in the picts_ml package.
"""


import sys
import obspy
import os
import shared_data_files.config as config  # user configuration


def check_dir_ready(path: str) -> None:
    """Checks that the specified directory exists and is empty.

    Args:
        path: the path to the directory.

    Returns:
        None if successful. Raises an error or exits the program if an issue occurs.
    """
    # check the directory exists
    if not os.path.exists(path):
        os.mkdir(path)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Something went wrong creating dir: {path}"
            )
    # check directory is empty
    if os.listdir(path):
        sys.exit(
            f"Check failed: The {path} directory already has files in it"
        )
    return


def get_relevant_streams(
    *,
    start_time: obspy.UTCDateTime,
    end_time: obspy.UTCDateTime,
    network: str,
    station: str,
) -> obspy.Stream | None:
    """Get obspy streams according that meet specified parameters.

    Args:
        start_time: The time the streams should start.
        end_time: The time the streams should stop.
        network: The two character network code.
        station: The four character station code.

    Returns:
        A stream if suitable files were found, otherwise None.
    """
    if start_time.julday == end_time.julday:
        # if start and end time are on the same day
        file_paths = get_files(
            network=network,
            station=station,
            year=str(start_time.year),
            jday=str(start_time.julday),
        )
        streams = obspy.Stream()
        for path in file_paths:
            streams += obspy.read(path, starttime=start_time, endtime=end_time)
    else:
        # if start and end time are on different days
        file_paths = get_files(
            network=network,
            station=station,
            year=str(start_time.year),
            jday=str(start_time.julday),
        )
        streams = obspy.Stream()
        for path in file_paths:
            streams += obspy.read(path, starttime=start_time)
        file_paths = get_files(
            network=network,
            station=station,
            year=str(end_time.year),
            jday=str(end_time.julday),
        )
        for path in file_paths:
            streams += obspy.read(path, endtime=end_time)
    if not streams:
        print(f"No streams found for station: {station}, time: {start_time}")
        return None
    # check that all streams have the same sampling rate
    for trace in streams:
        if trace.stats.sampling_rate != streams[0].stats.sampling_rate:
            print(f"Streams had different sampling rates for station: {station}, time: {start_time}")
            return None
    return streams


def get_files(*, network: str, station: str, year: str | int, jday: str | int) -> list[str | None]:
    """Returns a list of file paths for files that match the arguments.

    Args:
        network: The two character network code.
        station: The four character station code.
        year: The year, e.g. 2023.
        jday: The jul day, e.g. 254

    Returns:
        A list of paths to the files, or an empty list if no files are found.
    """
    network, station, year, jday = str(network), str(station), str(year), str(jday)
    file_list = list()
    for path in config.data_paths:
        location = os.path.join(path, year, network, station)
        for root, dirs, files in os.walk(location):
            for file in files:
                if not file.startswith(".") and file.endswith(jday):
                    file_list.append(os.path.join(root, file))
    return file_list
