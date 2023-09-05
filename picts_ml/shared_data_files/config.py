"""
This file contains configuration data that users will need to set themselves.
It is used by all scripts in the PICTS_ML repository. Detailed descriptions
for each setting are below.
"""


import pandas as pd
from pathlib import Path


# a list of strings giving full paths to the data from the seismometers
# this data can be split into different directories, but the format of each
# directory structure pointed to must be
# [year]/[network code]/[station code]/[sensor component]/[files with correct naming structure
# the file naming structure is [network code].[station code].00.[sensor component].[year].[julday]
# example of this setting:
# PICTS_DATA_PATHS = ["/run/media/dv1/Dave_T5/PICTS/PICTS_data/", "/run/media/dv1/Dave_T5/PICTS/PICTS_pullout/"]
# example of a single file path using this convention:
# /run/media/dv1/Dave_T5/PICTS/PICTS_data/2022/9J/B1BC/HHN.D/9J.B1BC.00.HHN.D.2022.218
data_paths = ["/run/media/dv1/Dave_T5/PICTS/PICTS_data", ]

# this is a csv file (commas / newlines but no spaces), located in the
# "shared_data_files" directory and containing information for the seismometer
# stations. It must have the following as first line (headings):
# lon,lat,name,network
# lon and lat are in decimal, using -ve and +ve convention. Name is 4 letter
# station code and network is 2 letter network code. The info is held as a pandas DataFrame
# example: station_info_name = "picts_network_info.csv"
station_info_name = "picts_network_info.csv"
# load the station info file (this part shouldn't need altering)
script_dir = Path(__file__).resolve().parent
file_path = script_dir / station_info_name
station_info = pd.read_csv(file_path, header=[0])

# Constants to be used for estimating P & S wave travel times (usually using TauPyModel)
# example: EARTH_RADIUS = 6363.133  # taken from https://rechneronline.de/earth-radius/ for lat = 57
# EARTH_FLATTENING = 0
earth_radius = 6363.133  # taken from https://rechneronline.de/earth-radius/ for lat = 57
earth_flattening = 0
