# PICTS_ML

Machine-learning related code and data I created whilst working on PICTS helping to research earthquakes in the North East of Scotland. The picts_ml package mainly focuses on training machine learning models that can assist in exploring seismometer data to find earthquakes and other waveforms of interest. A general workflow using this package might look like:

**pre-process data** ⟶ **generate training data** ⟶ **train ML model** ⟶ _use ML model to find interesting waveforms, identify earthquakes, pick P & S wave arrivals etc_


## Quickstart
Clone the repository with `git clone https://github.com/dgc-proton/PICTS_ML.git`, or alternatively click on the `Code` dropdown button at the top of the GitHub page ⟶ `Download ZIP` ⟶ extract the files to a suitable directory on your computer.

Configure the file paths located in `picts_ml/shared_data_files/config.py`.

Install the [requirements]( #requirements ).

The main scripts (a01... etc) have a Command Line Interface; so for example `python3 a01_pre_process_data.py -h` will give the CLI help / instructions for that script.

Source code for each script contains detailed documentation, including how the functions can be imported into your own scripts.

Located in appropriately named directories and files are some models already trained using the picts_ml scripts, and an ongoing events catalogue, amongst other things.


## Requirements

I recommend installing the requirements to a virtual environment (python venv has been tested, Anaconda or similar should work).

Required packages: [ObsPy]( https://github.com/obspy/obspy ), [pandas]( https://pandas.pydata.org/ ), [NumPy]( https://numpy.org/ ), [SeisBench]( https://github.com/seisbench/seisbench ), [Pyrocko]( https://github.com/pyrocko/pyrocko ), [PyTorch]( https://pytorch.org/ )
Optional packages: [matplotlib]( https://matplotlib.org/ )

`pip install -r requirements.txt` will install the tested versions of all required packages apart from pyrocko version 2023.06.29, which I recommend installing from source: <https://github.com/pyrocko/pyrocko>


## Background

I'm Dave Riley, the author of this repository and an electrical & mechanical engineering student at the University of Aberdeen. For 10 weeks during the summer of 2023 I have been working alongside Szymon Szymanski, together assisting [Dr Amy Gilligan]( https://amygilligan.wordpress.com ) with PICTS; Probing Into the Crust Through eastern Scotland. The key aim of PICTS is to uncover the role that the Highland Boundary Fault (HBF) has played in building Scotland using seismology.

Once the initial data exploration and fieldwork had been completed I spent my remaining time working with Szymon and Amy on processing and exploring the data. This is still a work in progress, but as my placement comes to an end I decided to select the parts of my work that I think will be most useful as the project continues and spend some time making them easier to access and use. There are two main aspects to this:

1. Gathering together the more useful machine-learning related code and associated data into a repository (this repository). Improve documentation, code comments and user interfaces so that people can quickly start using these tools as they are, make improvements to them or grab bits of code from them.
2. Work with Szymon to produce a poster which we will present at the BGA PGRiP 2023 conference, showing an overview of the project and the research that we have done so far. 

Please note that I have no background in geophysics or machine learning, so don't take for granted that anything in this repository is correct! I started studying these subjects in earnest in the months leading up to my placement, and have had help and support from Amy and from Szymon throughout the placement.

Overall I've found these subjects and the experience of working on PICTS very interesting and would highly recommend it!


## Overview of the PICTS_ML Repository

Much more detail is contained within the source code; I've tried to ensure that there are instructions for use at the top of each file as well as sufficient comments, descriptive docstrings and type-hints for each function.


### 01: Data Pre-Processing

`a01_pre_process_data.py` takes a csv file with known event times and locations, and outputs a csv catalogue which details the times of P & S waves from the event arriving at each seismometer station as well as metadata in the required format for later scripts to use. Picks are done manually, with automation assistance giving the user estimated arrival times and ensuring the relevant section of the trace is displayed on which to make the picks.


### 02: Generating Training Data

`a02_generate_training_data.py` takes a catalogue of events and picks (such as one produced by `a01_pre_process_event_data.py`), and transform it into a `metadata.csv` file and a `waveforms.hdf5` file suitable for using to train an ML picking or detecting model using Seisbench.


### 03: Training (or Re-training) Models

`a03_train_a_model.py` trains a model, giving the option to load a pre-trained model which is then retrained. The data to perform the training needs to be in the format that is generated by `a02_generate_training_data.py`, or with a little code alteration datasets in the normal Seisbench format could also be used.


### Pre-Processed Events Catalogue

The `pre-processed_events_catalogue directory` contains at least one catalogue of events with manual pick times for P & S wave arrivals at PICTS stations. The initial event details were obtained from the BGS (British Geological Survey) database, and the plan is to add events to this file from other sources as well.

Notes have been made where traces weren't available or where picks were too difficult for me to make. When considering the picks I have made, please remember that I had no experience picking before this so depending on your use-case you may want to re-pick some of them. The files have been created using `a01_pre_process_data.py` and so can be easily turned into training data.


### Trained Models

Contains some trained and ready to use models created with these scripts, along with details of how they were trained.


## FAQ

**Q** Why is the naming of some directories or files a bit weird? What are the various `__init__.py` files for?

**A** I've quickly tried to make it easy for someone to import functions they may wish to use into their own code. Some of these also facilitate the internal imports that this codebase uses. If you mess with these then you may break imports!

**Q** I've spotted a bug / I'd like to give you some feedback / you have no idea what you're doing, and I can explain where you went wrong / I want to ask something. How can I contact you?

**A** Although I'm not going to actively maintain the repository, I'd be happy to respond when I get chance if you raise an issue on GitHub [github.com/dgc-proton/PICTS_ML]( https://github.com/dgc-proton/PICTS_ML ) or drop me an email [d.riley.22@abdn.ac.uk]( d.riley.22@abdn.ac.uk ).

**Q** Is there data missing from a csv generated by one of these scripts?

**A** Please expand the column and click into the cells to check; some spreadsheet programs don't always display content of some columns (usually the comments column) straight away.


## Acknowledgements

The Research Experience Placement I am on is run by QUADRAT (Queen’s University Belfast & Aberdeen Doctoral Research and Training) with funding for my wages and expenses provided by the National Environment Research Council. I would like to take this opportunity to thank everyone involved, especially Amy and Szymon who have been fantastic to work with and from whom I have learned a lot.

The PICTS project is supported by funding from a Royal Society of Edinburgh Small Grant, and is a collaboration with BGS Seismology.

The following open source projects were key to this work: [Seisbench]( https://seisbench.readthedocs.io ) [Obspy]( https://docs.obspy.org/ ) [Pyrocko]( https://pyrocko.org/ )

Josh Starmer's [StatQuest]( https://statquest.org/ ) really helped me to begin to understand some aspects of machine-learning.


(c) Copyright 2023, Dave Riley