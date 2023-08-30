"""
Description:
This script generates metadata.csv & waverforms.hdf5 files from pre-processed data.
The pre-processed data needs to be of a format similar to that produced by
a01_pre_process_data.py. These generated files are suitable to use for training
machine-learning models using the script a03_train_a_model or many modules of Seisbench.
With little modifications to this script the files can be tweaked to suit many training
requirements.
Some of this code was adapter from a notebook from the authors of Seisbench:
https://colab.research.google.com/github/seisbench/seisbench/blob/main/examples/03b_creating_a_dataset.ipynb

Usage:
Launch the program from the command line with no arguments to get the help text.

Alternatively import the generate_training_data function into another script to use it.
The simplest way to do this is to copy the picts_ml package into the same directory as
your new script, and then at the top of your new script:
from picts_ml.a02_generate_training_data import generate_training_data

Notes:
The log file only contains certain logs that I am able to capture; most warnings from
the Seisbench modules go to stdout and I haven't found a neat solution to log them yet.

The methods of picking the wave arrivals for the metadata and the train / dev /
test split could probably be improved.
"""


import argparse
from datetime import datetime

import obspy
from obspy import UTCDateTime
import pandas as pd
import numpy as np
import seisbench.data as sbd
import seisbench.util as sbu

import shared_data_files.config as config  # user config
from shared_data_files.shared_functions import get_relevant_streams, check_dir_ready


# output file paths
METADATA_PATH = "02_generate_training_data_outputs/metadata.csv"
WAVEFORMS_PATH = "02_generate_training_data_outputs/waveforms.hdf5"
LOGFILE_PATH = "02_generate_training_data_outputs/generator.log"
# time to pad around the P & S arrival stream
TIME_BEFORE_P_ARRIVAL = 60 * 5
TIME_AFTER_S_ARRIVAL = 60 * 5


def generate_training_data(preprocessed_data_path: str) -> None:
    """Reads a pre-processed data file and outputs training data files.

    The pre-processed data file must be of a similar format to those produced by
    a01_pre_preocess_data.py. This function will output a waveforms.hdf5 file, a
    metadata.csv file and a log file into the generated_training_data folder.

    Args:
        preprocessed_data_path: The path to the pre-processed data file.

    Returns:
        None.
    """
    # check that the output directory exists and has no files already in it
    check_dir_ready("generator_outputs")
    log_file("generation started")
    # create the random number generator used for test / train / validate split
    r_num_gen = np.random.default_rng(53485178)
    # load events catalogue
    events = pd.read_csv(preprocessed_data_path, header=[0])

    # writer for the files
    with sbd.WaveformDataWriter(METADATA_PATH, WAVEFORMS_PATH) as writer:
        writer.data_format = {"dimension_order": "CW", "component_order": "ZNE"}

        # iterate over the events
        for event_index in events.index:
            # iterate over the stations
            for station_index in config.station_info.index:
                station_name = (config.station_info.loc[station_index, "name"],)
                p_column = station_name[0] + "_p_arrival_time_man_picked"
                s_column = station_name[0] + "_s_arrival_time_man_picked"
                try:
                    p_arrival_time = UTCDateTime(events.loc[event_index, p_column])
                    s_arrival_time = UTCDateTime(events.loc[event_index, s_column])
                except TypeError:
                    # the file doesn't have a pick, skip to the next station
                    continue
                # get the relevant streams
                # NOTE that waves occuring over midnight have not been accounted for
                stream_start = p_arrival_time - TIME_BEFORE_P_ARRIVAL
                stream_end = s_arrival_time + TIME_AFTER_S_ARRIVAL
                streams = get_relevant_streams(
                    start_time=stream_start,
                    end_time=stream_end,
                    network=config.station_info.loc[station_index, "network"],
                    station=config.station_info.loc[station_index, "name"],
                )
                if not streams:
                    log_file(
                        f"No streams found for station: {station_name}, event time: {events.loc[event_index, 'utcdate']}"
                    )
                    continue
                # check that all streams have the same sampling rate
                for trace in streams:
                    if trace.stats.sampling_rate != streams[0].stats.sampling_rate:
                        log_file(
                            f"Streams had different sampling rates (so not written) for station: {config.station_info.loc[station_index, 'name']}, event time: {events.loc[event_index, 'utcdate']}"
                        )
                        continue
                # get streams formatted for writing
                actual_t_start, data, _ = sbu.stream_to_array(
                    streams, component_order=writer.data_format["component_order"]
                )
                # put metadata into a dictionary ready for writing
                metadata = compose_metadata(
                    event=events.loc[event_index],
                    station_info=config.station_info.loc[station_index],
                    sampling_rate=streams[0].stats.sampling_rate,
                    p_travel_time=None,
                    start=actual_t_start,
                    rng=r_num_gen,
                    p_arrival=p_arrival_time,
                    s_arrival=s_arrival_time,
                )
                # add data to the writer
                writer.add_trace(metadata, data)
    log_file("Finished")


def compose_metadata(
    *,
    event: pd.Series,
    station_info: pd.Series,
    sampling_rate: int,
    p_travel_time: int | None,
    start: obspy.UTCDateTime,
    rng: np.random.default_rng,
    p_arrival: obspy.UTCDateTime,
    s_arrival: obspy.UTCDateTime,
) -> dict[str, str | None | float | int]:
    """Composes metadata required for writing the metadata.csv and waveforms.hdf5 files
    into a dictionary.

    Args:
        event: The event information from the pre-processed data.
        station_info: The station information as laid out in config.py
        sampling_rate: The sampling rate of the waveforms (all waveforms must have the
                       same sampling rate).
        p_travel_time: The travel time in seconds of the P waves. Only used by some
                       models so can be None.
        start: The start time of the stream.
        rng: A suitable numpy random number generator instance to use for selecting
             the data split.
        p_arrival: The time of P wave arrival.
        s_arrival: The time of S wave arrival.

    Returns:
        A dict containing the metadata.
    """
    rand_num = int(rng.integers(low=0, high=10, size=1))
    # TODO consider more optimal way to split data
    if rand_num <= 1:
        chosen_split = "test"
    elif rand_num <= 3:
        chosen_split = "dev"
    else:
        chosen_split = "train"
    p_sample_arrival = (p_arrival - start) * sampling_rate
    s_sample_arrival = (s_arrival - start) * sampling_rate
    return {
        "station_network_code": station_info["network"],
        "station_code": station_info.name,
        "trace_channel": "HH",
        "station_latitude_deg": station_info["lat"],
        "station_longitude_deg": station_info["lon"],
        "station_elevation_m": 0,  # NOTE this is populated in the STEAD dataset
        "trace_p_arrival_sample": p_sample_arrival,
        "trace_p_status": None,  # NOTE this is populated in the STEAD dataset
        "trace_p_weight": None,  # NOTE this is populated in the STEAD dataset
        "path_p_travel_sec": p_travel_time,
        "trace_s_arrival_sample": s_sample_arrival,
        "trace_s_status": None,  # NOTE this is populated in the STEAD dataset
        "trace_s_weight": None,  # NOTE this is populated in the STEAD dataset
        "source_id": None,  # NOTE this is populated in the STEAD dataset
        "source_origin_time": event["utc_datetime"],
        "source_origin_uncertainty_sec": None,  # NOTE this is populated in the STEAD dataset
        "source_latitude_deg": event["lat"],
        "source_longitude_deg": event["lon"],
        "source_error_sec": None,  # NOTE this is populated in the STEAD dataset
        "source_gap_deg": None,  # NOTE this is populated in the STEAD dataset
        "source_horizontal_uncertainty_km": None,  # NOTE this is populated in the STEAD dataset
        "source_depth_km": event["depth"],
        "source_depth_uncertainty_km": None,  # NOTE this is populated in the STEAD dataset
        "source_magnitude": event["ML"],
        "source_magnitude_type": "ml",
        "source_magnitude_author": None,  # NOTE this is populated in the STEAD dataset
        "source_mechanism_strike_dip_rake": None,  # NOTE this is populated in the STEAD dataset
        "source_distance_deg": None,  # NOTE this is populated in the STEAD dataset
        "source_distance_km": None,  # NOTE this is populated in the STEAD dataset
        "path_back_azimuth_deg": None,  # NOTE this is populated in the STEAD dataset
        "trace_snr_db": None,  # NOTE this is populated in the STEAD dataset
        "trace_coda_end_sample": None,  # NOTE this is populated in the STEAD dataset
        "trace_start_time": start,
        "trace_category": None,  # NOTE this is populated in the STEAD dataset
        "trace_name": None,  # NOTE this is populated in the STEAD dataset
        "split": chosen_split,
        "trace_name_original": None,  # NOTE this is populated in the STEAD dataset
        "trace_sampling_rate_hz": sampling_rate,
    }


def log_file(message: str) -> None:
    """Adds timestap & message to the logfile."""
    with open(LOGFILE_PATH, "a") as file:
        log_message = f"{datetime.now()}:    {message}\n"
        file.write(log_message)
    print(log_message)
    return


if __name__ == "__main__":
    # setup the argument parser
    parser = argparse.ArgumentParser(
        description="A program that takes a pre-processed data file as an input from the user, and then outputs a "
        " metadata.csv & waverforms.hdf5 file suitable for training a machine learning model. Files are output "
        "to 02_generate_training_data_outputs. See documentation (in code and in README) for more information."
    )
    parser.add_argument(
        "pre_processed_data_path",
        help="The path / filename of the pre-processed data to use in " "generating training data.",
    )
    args = parser.parse_args()
    generate_training_data(args.pre_processed_data_path)
