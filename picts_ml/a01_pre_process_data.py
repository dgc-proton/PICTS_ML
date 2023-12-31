"""
Description:
This script pre-processes event data and outputs a new csv file. At present it is setup
for an input csv file with information similar to that which is provided by the BGS
event database (this mapping behaviour can easily be altered to accomodate other files
by editing the functions df_columns & dict_general_event_info.  This script will then:
- Add a utc date column in obspy format
- Estimate P & S wave arrivals at stations using Obspy TauPyModel
- Manual picking of P & S wave arrivals, with the console (not GUI) giving helpful info
- (optional, not reccomended) deep-denoiser ML based denoising can be applied to assist
  manual picks; turn this on by changing the DEEP_DN constant below the import statements.

Previously just the estimated P & S wave arrivals could be used without the step of
manually picking them, however this was not sufficiently accurate and produced inferior
models. It would be easy to modify the script to do this again or to use a different
estimation method, eliminating the need for so many manual picks.


Usage:
Launch the program from the command line with no arguments to get the help text.

Alternatively import the pre_process_data function into another script to use it. The
simplest way to do this is to copy the picts_ml package into the same directory as your
new script and then at the top of your new script:
from picts_ml.a01_pre_process_data import pre_process_data

Picking:
- Double clicking brings into pick mode. Then single click to mark, and
  press F1 to designate as P wave or F2 to designate as S wave.
- Closing the window will send the pick data back to this program and prompt asking if
  you want to add any comments (press enter to not add a comment).
- Once all picks have been made for a particular event the data will be written to file,
  and you're given the option to continue or quit.
- Quitting 'finalises' the file; to continue where you left off use the import function
  and a copy of the event data where you have manually removed events that have already
  been picked.
"""


import sys
import os
from datetime import datetime
import argparse

import obspy
from obspy.taup import TauPyModel, taup_geo
import seisbench
import seisbench.models as sbm
from pyrocko import obspy_compat
import pandas as pd

import shared_data_files.config as config  # user defined config options
from shared_data_files.shared_functions import get_relevant_streams, get_files


# setup obspy-pyrocko compatability
obspy_compat.plant()


def pre_process_data(*, use_deep_dn: bool = False, file_path: str, add_existing: list[str] | None = None) -> None:
    """This is the main function, which carries out the data pre-processing.

    Args:
        use_deep_dn: True if deep denoiser machine learning denoiser to be applied to
                     streams before P & S waves are manually picked.
        file_path: The path to the file which contains details of the events. Format is
                   to be csv and similar to BGS data.
        add_existing: A list of paths of already pre-processed files which are to be
                      added to the output file, or None if no files to add.
    Returns:
        None.
    """
    filename = f"preprocessed_event_data_{datetime.now()}.csv"
    path = os.path.join("01_pre_process_data_outputs", filename)
    events_file = pd.read_csv(file_path, header=[0])
    # check that we won't overwrite data
    if os.path.isfile(path):
        sys.exit(f"file {path} already exists.... exiting")
    # create blank dataframe
    master_df: pd.DataFrame = pd.DataFrame(columns=df_columns())
    # add specified files of already pre-processed data to the df
    if add_existing:
        to_add = list()
        for existing_file in add_existing:
            to_add.append(pd.read_csv(existing_file, header=[0]))
        master_df = add_existing_to_df(df=master_df, to_add=to_add)
    # pre-process the new data files & add to the df, saving during processing
    print(
        process_pick_save(
            df=master_df, save_path=path, file=events_file, use_deep_dn=use_deep_dn, originating_file_name=file_path
        )
    )


def df_columns(*, include_arrival_times=True) -> dict[str, list[str] | str | None]:
    """Returns a dict to map columns from the old file to the new dataframe / file.

    Args:
        include_arrival_times: If false then the columns related to wave arrival times
                               will not be included in the dictionary. Default: True.

    Returns:
        A dict of column names mapping {"new_column_name": "old_file_column_name",}.
        Note that exact mapping behaviour is defined by other functions, this dictionary
        just details which columns are related.
    """
    columns: dict = {
        "utc_datetime": ["yyyy-mm-dd", "hh:mm:ss.ss"],
        "lat": "lat",
        "lon": "lon",
        "depth": "depth",
        "ML": "ML",
        "RMS": "RMS",
        "intensity": "intensity",
        "locality_1": "locality_1",
        "locality_2": "locality_2",
        "locality_3": "locality_3",
        "name_of_originating_file": None,
        "comments": "",
    }
    if include_arrival_times:
        # add the p & s wave arrival column names
        for col_name in arrival_time_columns():
            columns[col_name] = None
    return columns


def arrival_time_columns() -> list[str]:
    """Returns the p & s wave arrival time column names."""
    column_names: list = list()
    for station_name in config.station_info.loc[:, "name"]:
        column_names.append(f"{station_name}_p_arrival_time_man_picked")
        column_names.append(f"{station_name}_s_arrival_time_man_picked")
    return column_names


def process_pick_save(
    *, df: pd.DataFrame, save_path: str, file: pd.DataFrame, use_deep_dn: bool = False, originating_file_name: str
) -> str:
    """Carries out the data pre-processing including manually picking wave arrivals.

    Loops over the events in the file. For each event, a dictionary containing relevant
    data is built. This includes manual P & S wave arrival picks, user comments and
    other data needed to train an ML picking model. After every station has been looped
    over and the data / picks added to the dict, the dict is added to the dataframe and
    the dataframe is saved. The first save created the pre-processed data csv file,
    subsequent saves overwrite this file. The user has the option to quite after a save
    or to continue. Once the all events have been processed a success message is returned.

    Args:
        df: A dataframe with the required columns already added, and any pre-existing
            data already added.
        save_path: The path inc filename to save the dataframe to.
        file: The events data from the events data file.
        use_deep_dn: Bool indicating if deep denoiser ML denoising should be used on
                     the streams prior to making manual picks. Default: False.
        originating_file_name: The name of the file that the data came from.

    Returns:
        A string indicating that all events in the file have been processed.
    """
    taupy_model = TauPyModel(model="ak135")
    if use_deep_dn:
        dd_model = sbm.DeepDenoiser.from_pretrained("original")
    # iterate over the events from the new File
    for new_event_index in file.index:
        # add general event information to an event information dictionary
        event_info = dict_general_event_info(
            file=file, file_event_index=new_event_index, originating_file_name=originating_file_name
        )
        # iteraterate over the stations doing manual p & s wave picks
        for station_name in config.station_info.loc[:, "name"]:
            # estimate arrival times
            event_time = obspy.UTCDateTime(event_info["utc_datetime"])
            arrivals = estimate_ps_arrivals_taupy(
                model=taupy_model,
                event=event_info,
                # apologies that the below pandas code is so ugly
                # TODO make pandas code more readable
                station_lat=float(config.station_info.loc[config.station_info.name == station_name, "lat"].iloc[0]),
                station_lon=float(config.station_info.loc[config.station_info.name == station_name, "lon"].iloc[0]),
            )
            if not arrivals:
                # if arrivals have failed to calculate assign a stream window based on the event time
                print(f"Arrivals have failed to calculate for {event_info['utc_datetime']}")
                stream_start_time = event_time - (60 * 1)
                stream_end_time = event_time + (60 * 8)
            else:
                # if arrivals have calculated then assign a stream window based on the estimated arrival times
                stream_start_time = event_time - (60 * 1)
                stream_end_time = event_time + arrivals[1].time + (60 * 1)
            # get relevant streams for the event
            streams = get_relevant_streams(
                start_time=stream_start_time,
                end_time=stream_end_time,
                network=str(config.station_info.loc[config.station_info.name == station_name, "network"].iloc[0]),
                station=station_name,
            )
            if not streams:
                # skip to the next station in loop if no streams found for this station at the event time
                event_info[f"{station_name}_p_arrival_time_man_picked"] = "no streams available"
                event_info[f"{station_name}_s_arrival_time_man_picked"] = "no streams available"
                continue
            # get manual picks
            # print information to help with making the manual picks
            if arrivals:
                print(
                    f"for station: {station_name} event: {event_time} estimates were ....\n"
                    f"p: {event_time + arrivals[0].time}\ns: {event_time + arrivals[1].time}"
                )
            else:
                print(f"for station: {station_name} event: {event_time} estimates failed to calculate.")
            if use_deep_dn:
                p_wave_marker, s_wave_marker, picks_made = manual_pick(dd_model=dd_model, stream=streams)
            else:
                p_wave_marker, s_wave_marker, picks_made = manual_pick(stream=streams)
            # capture any picker comments
            comments = input("Enter any comments here: ")
            if comments:
                event_info["comments"] += f"| {station_name}: {comments} |\n"
            # if no picks were made note this then move to next station
            if not picks_made:
                event_info[f"{station_name}_p_arrival_time_man_picked"] = "difficult pick"
                event_info[f"{station_name}_s_arrival_time_man_picked"] = "difficult pick"
                continue
            print(f"p_wave: {p_wave_marker}\ns_wave: {s_wave_marker}\n*****************************\n")
            event_info[f"{station_name}_p_arrival_time_man_picked"] = p_wave_marker
            event_info[f"{station_name}_s_arrival_time_man_picked"] = s_wave_marker
        # save the df after each event
        df = pd.concat([df, pd.DataFrame.from_records([event_info])], ignore_index=True)
        df.to_csv(save_path)
        response = input("Work saved! Enter 'q' to quit, or anything else to continue: ")
        if response == "q":
            sys.exit()


def manual_pick(
    *, dd_model: seisbench.models.deepdenoiser.DeepDenoiser | None = None, stream: obspy.Stream
) -> tuple[obspy.UTCDateTime | None, obspy.UTCDateTime | None, bool]:
    """Get user to manually pick P & S waves.

    Args:
        dd_model: an instance of deep denoiser model or None if deep denoiser is not to
                  be used. Default: None.
        stream: the stream on which the manual picks are to be made.

    Returns:
        A tuple of the P wave pick, the S wave pick, and a bool indicating if picks
        were made or not.
    """
    # keep bringing up the manual pick GUI until an exit condition is reached
    while True:
        # bring up manual pick GUI on either the unaltered stream or on a deep denoised stream as appropriate
        if dd_model:
            dd_stream = dd_model.annotate(stream)
            picks = dd_stream.snuffle()
        else:
            picks = stream.snuffle()
        try:
            # get the picks returned by the GUI
            p_wave_marker = obspy.UTCDateTime(picks[1][0].tmin)
            s_wave_marker = obspy.UTCDateTime(picks[1][1].tmin)
        except IndexError:
            # if no picks were made
            return None, None, False
        # check that p & s picks are the right way around
        if p_wave_marker < s_wave_marker:
            return p_wave_marker, s_wave_marker, True
        # p and s picks are the wrong way around - repeat the pick
        print(
            "Please repeat the pick; p waves must always arrive before s waves." "Make sure you pick P before picking S"
        )


def dict_general_event_info(
    *, file: pd.DataFrame, file_event_index: int, originating_file_name: str
) -> dict[str, obspy.UTCDateTime | str]:
    """Returns a dict of general event information taken from the specified file.

    Uses the df_columns function to provide the name of the columns in the event file
    that maps to the dict key: value. This function then checks if there is a special
    case for processing, and if not uses a standard mapping as shown in the code.
    to map file headings to dictionary keys.

    Args:
        file: The data from the events file.
        file_event_index: The index of the event that is to be used to populate the dict.

    Returns:
        A dictionary containing all information for the event that is to be written to
        the pre-processed data file, apart from information to do with P & S wave
        arrival picks or comments.
    """
    event_info = dict()
    for df_col_name, file_col_name in df_columns(include_arrival_times=False).items():
        if df_col_name == "utc_datetime":
            event_info[df_col_name] = obspy.UTCDateTime(
                file.loc[file_event_index, file_col_name[0]] + file.loc[file_event_index, file_col_name[1]]
            )
            continue
        if df_col_name == "name_of_originating_file":
            event_info[df_col_name] = originating_file_name
            continue
        if df_col_name == "comments":
            event_info[df_col_name] = ""
            continue
        # if file_col_name is None at this stage then something has gone wrong
        if file_col_name is None:
            raise ValueError(
                f"file_col_name is None, but no special handling cases have been programmed for"
                f"df_col_name={df_col_name} in function add_new_to_df"
            )
        # if made it to this point then there should be a valid standard mapping provided by the df_columns function
        event_info[df_col_name] = file.loc[file_event_index, file_col_name]
    return event_info


def estimate_ps_arrivals_taupy(
    *, model: obspy.taup.TauPyModel, event: dict, station_lat: float, station_lon: float
) -> None | obspy.taup.tau.Arrivals:
    """Returns P & S wave arrival estimates, or None if they fail to calculate.

    Args:
        model: An instance of TauPyModel.
        event: The event information dictionary, already populated with event location
               and depth.
        station_lat: station latitude
        station_lon: station longitude

    Returns:
        None if fails to calculate (sometimes even with valid parameters this will
        happen, I haven't managed to find the reason why the TauPyModel ocassionally
        fails). Otherwise returns an Arrivals object. Hint: p-wave arrival time is
        obspy.taup.tau.Arrivals[0].time, s-wave is obspy.taup.tau.Arrivals[0].time
    """
    # get angular distance between event and station
    ang_distance = taup_geo.calc_dist_azi(
        event["lat"], event["lon"], station_lat, station_lon, config.earth_radius, config.earth_flattening
    )[0]
    # estimate P/S arrival delay
    arrivals = model.get_travel_times(
        source_depth_in_km=event["depth"], distance_in_degree=ang_distance, phase_list=["P", "S"]
    )
    return arrivals


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


def add_existing_to_df(*, df: pd.DataFrame, to_add: list[pd.DataFrame]) -> pd.DataFrame:
    """Adds the data from already pre-processed files to the dataframe.

    Args:
        df: The df which the data is to be added to. Must already have correct columns.
        to_add: A list of the dataframes to add.

    Returns:
        Dataframe with all data in it.
    """
    # check that data all has same headings
    for additional_df in to_add:
        if not df.columns == additional_df.columns:
            raise ValueError(
                "Some of the files that were requested to be added do not"
                "have the correct column names, cannot continue operation."
            )
    for additional_df in to_add:
        df = pd.concat([df, additional_df], ignore_index=True)
    return df


if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description="A program that takes an event file as an input, gets manual picks "
        "from the user, and then outputs a csv file suitable for both training"
        " a machine learning model as well as general documentation. Please "
        " see documentation (in code and in README) for more information."
    )
    parser.add_argument("events_file", help="The file containing events in a format similar to the BGS database.")
    parser.add_argument(
        "--use_deep_denoiser",
        help="Process streams with the machine learning deep denoiser tool " "prior to picking P & S arrivals.",
        action="store_true",
    )
    parser.add_argument(
        "--existing_file",
        help="Any existing pre-processed files that you want to add to the resulting csv.",
        type=argparse.FileType("r"),
        nargs="+",
    )
    # Parse arguments provided by user
    args = parser.parse_args()
    # call the main function with arguments
    pre_process_data(
        use_deep_dn=args.use_deep_denoiser, file_path=args.events_file, add_existing=args.existing_file
    )
