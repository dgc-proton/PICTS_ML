"""
Description:
This scripts takes training data in the form of a waveforms.hdf5 and metadata.csv file,
and trains a machine learning picker model using this training data along with the
training parameters given. It is able to train a model from 'scratch' (i.e. randomly
pre-weighted), or it can load a model that has already been trained on a dataset and
then re-train it. 
A lot of the code was adapted from a notebook written by the authors of Seisbench:
https://colab.research.google.com/github/seisbench/seisbench/blob/main/examples/03a_training_phasenet.ipynb


Usage:
Launch the program from the command line with no arguments to get the help text.

Alternatively import the retrain_a_model function into another script to use it.
The simplest way to do this is to copy the picts_ml package into the same directory as
your new script, and then at the top of your new script:
from picts_ml.a03_train_a_model import retrain_a_model

Notes:
This script probably has a lot of room for improvement. As of August 2023 initial tests
indicate that PhaseNet models pretrained on STEAD and retrained on a limited amount of
PICTS data may outperform the same model without retraining. I'm currently pre-processing
more PICTS data so that a more thorough evaluation of the models can be done. -Dave
"""

import os
import sys
from datetime import datetime
import argparse

import seisbench
import seisbench.models as sbm
import seisbench.data as sbd
import seisbench.generate as sbg
from seisbench.util import worker_seeding
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
def visualisation_on():
    import matplotlib.pyplot as plt

import shared_data_files.config as config


def retrain_a_model(
    *,
    model_type: str,
    model_pretraining: str,
    saving_name: str,
    save_version: int = 1,
    save_path="retrained_models",
    training_data: str,
    epochs: int = 100,
    pg_sigma: int = 15,
    on_hpc: bool = False,
    show_trg_example: bool = False,
    learning_rate: float = 1e-2,
) -> None:
    """Trains a model according to the specified parameters, then saves it ready for use.

    Args:
        model_type: The type of model, all lowercase, e.g. phasenet
        model_pretraining: The pretraining for the model, or None for a randomly weighted model
        saving_name: The name to save the trained model as
        save_version: The version number to save the model as
        save_path: The path to save models on
        training_data: The location of the training data directory containing waveforms.hdf5 and metadata.csv
        epochs: Training epochs
        pg_sigma: Sigma value for predictive labeller
        on_hpc: Set True to load model to Cuda 
        show_trg_example: Display graphs of a randomly selected training example during training
        learning_rate: Learning rate for training

    Returns:
        None
    """
    # check that the name the model will be saved as is not already in use
    _check_saving_name(saving_path=save_path, name=saving_name)
    # cast the save version to str for later saving (argument is int to make sure valid version number supplied)
    save_version  = str(save_version)
    
    # create the pre-trained model
    model = _select_model(model_name=model_type, model_pretrain=model_pretraining)
    if on_hpc:
        model.cuda()

    # load the data for training
    data = sbd.WaveformDataset(
        training_data, sampling_rate=100, missing_componens="pad"
    )

    # split the dataset
    train, dev, test = data.train_dev_test()

    # create generators for training and validation
    phase_dict = dict()
    for station_name in config.station_info.loc[:, "name"]:
        phase_dict[f"{station_name}_p_arrival_time_man_picked"] = "P"
        phase_dict[f"{station_name}_s_arrival_time_man_picked"] = "S"

    train_generator = sbg.GenericGenerator(train)
    dev_generator = sbg.GenericGenerator(dev)

    augmentations = [
        sbg.WindowAroundSample(
            list(phase_dict.keys()),
            samples_before=3000,
            windowlen=6000,
            selection="random",
            strategy="variable",
        ),
        sbg.RandomWindow(windowlen=3001, strategy="pad"),
        sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
        sbg.ChangeDtype(np.float32),
        sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=pg_sigma, dim=0),
    ]

    train_generator.add_augmentations(augmentations)
    dev_generator.add_augmentations(augmentations)

    # visualise a training example
    if show_trg_example:
        sample = train_generator[np.random.randint(len(train_generator))]
        fig = plt.figure(figsize=(15, 10))
        axs = fig.subplots(
            2, 1, sharex=True, gridspec_kw={"hspace": 0, "height_ratios": [3, 1]}
        )
        axs[0].plot(sample["X"].T)  # plot waveforms
        axs[1].plot(sample["y"].T)  # plot labels
        plt.show()

    # create pytorch loaders
    batch_size = 256
    num_workers = 4  # The number of threads used for loading data
    train_loader = DataLoader(
        train_generator,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=worker_seeding,
    )
    dev_loader = DataLoader(
        dev_generator,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=worker_seeding,
    )

    # training parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # do the training
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        _train_loop(train_loader, model, optimizer)
        _test_loop(dev_loader, model)

    # save the model
    path = os.path.join(save_path, saving_name)
    model.save(path, version_str=save_version)
    info_path = os.path.join(path, "model_info.txt")
    info = (
        f"Date created: {datetime.now()}\nName: {saving_name}\n"
        f"Model type: {model_type} '{model_pretraining}'\n"
        f"Training data: {training_data}\nEpochs: {epochs}\n"
        f"Sigma: {pg_sigma}"
    )
    with open(info_path, "w") as file:
        file.write(info)
    return


def _loss_fn(y_pred, y_true, eps=1e-5):
    """Loss function, taken from the Seisbench example notebook."""
    # vector cross entropy loss
    h = y_true * torch.log(y_pred + eps)
    h = h.mean(-1).sum(-1)  # Mean along sample dimension and sum along pick dimension
    h = h.mean()  # Mean over batch axis
    return -h


def _train_loop(dataloader, model, optimizer):
    """Training loop, taken from the Seisbench example notebook."""
    size = len(dataloader.dataset)
    for batch_id, batch in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(batch["X"].to(model.device))
        loss = _loss_fn(pred, batch["y"].to(model.device))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_id % 5 == 0:
            loss, current = loss.item(), batch_id * batch["X"].shape[0]
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def _test_loop(dataloader, model):
    """Test loop, taken from the Seisbench example notebook."""
    num_batches = len(dataloader)
    test_loss = 0

    model.eval()  # close the model for evaluation

    with torch.no_grad():
        for batch in dataloader:
            pred = model(batch["X"].to(model.device))
            test_loss += _loss_fn(pred, batch["y"].to(model.device)).item()

    model.train()  # re-open model for training stage

    test_loss /= num_batches
    print(f"Test avg loss: {test_loss:>8f} \n")


def _select_model(model_name: str, model_pretrain: str | None = None) -> seisbench.models.base.WaveformModel:
    """Returns the Seisbench model specified.
    
    Args:
        model_name: The name of the model type to use, all in lower-case.
        model_pretrain: The model pretraining type (usually a dataset name). If none
                        the model will be randomly weighted.

    Returns:
        An instance of the specified model.

    Raises:
        ValueError with description if model or pretraining are not valid.
    """
    try:
        match model_name:
            case "basicphaseae":
                if model_pretrain:
                    return sbm.BasicPhaseAE.from_pretrained(model_pretrain)
                else:
                    return sbm.BasicPhaseAE()
            case "cred":
                if model_pretrain:
                    return sbm.CRED.from_pretrained(model_pretrain)
                else:
                    return sbm.CRED()
            case "dpp":
                if model_pretrain:
                    return sbm.DeepPhasePick(model_pretrain)
                else:
                    return sbm.DeepPhasePick()
            case "eqtransformer":
                print("\n*****\nWARNING\n*****\nBugs with training EQTransformer were encountered during testing "
                      "August 2023; they appeared to be within Seisbench, investigation will continue if time permits")
                if model_pretrain:
                    return sbm.EQTransformer.from_pretrained(model_pretrain)
                else:
                    return sbm.EQTransformer()
            case "gpd":
                if model_pretrain:
                    return sbm.GPD.from_pretrained(model_pretrain)
                else:
                    return sbm.GPD()
            case "phasenet":
                if model_pretrain:
                    return sbm.PhaseNet.from_pretrained(model_pretrain)
                else:
                    return sbm.PhaseNet()
            case "phasenetlight":
                if model_pretrain:
                    return sbm.PhaseNetLight.from_pretrained(model_pretrain)
                else:
                    return sbm.PhaseNetLight()
            case "pickblue":
                raise ValueError(
                    "Error in _select_model(): PickBlue is intended for the "
                    "ocean! Model not created."
                )
            case _:
                raise ValueError(
                    f"model: ({model_name}) was not recognised "
                    f"(it may not have been implemented in this code yet)."
                )
                
    except ValueError:
        raise ValueError(
            f"pretraining ({model_pretrain}) was not recognised, it may not "
            f"have been implemented in this code yet). Model not created."
        )


if __name__ == "__main__":
    # setup the argument parser for CLI
    parser = argparse.ArgumentParser(
        description="A program that takes one set of training data as an input from directory 03_train_a_model_inputs, "
        "along with training parameters, and then outputs a trained ML model to the directory 03_tran_a_model_outputs")
    parser.add_argument("training_data", help="The directory containing the training data (waveforms.hdf5 & metadata.csv)")
    parser.add_argument("saving_name", help="The name to save the trained model as.")
    parser.add_argument("--type", help="The model type (note: all lowercase), default=phasenet. See Seisbench documentation for all options.", default="phasenet")
    parser.add_argument("--pretraining", help="Dataset for pretraining. Default=None (a randomly pre-weighted model). Reccomended=stead. See Seisbench documentation for options.", default=None)
    parser.add_argument("--epochs", help="Training epochs. Default=100", default=100, type=int)
    parser.add_argument("--sigma", help="Sigma value for probabalistic labeller. Default=15", default=15, type=int)
    parser.add_argument("--cuda", help="Activate CUDA GPU acceleration.", action="store_true")
    parser.add_argument("--visualise_trg_example", help="Display a graph of a single randomly selected training sample during training.", action="store_true")
    parser.add_argument("--learning_rate", help="The learning rate to apply to training. Default=1e-2", default=1e-2. type=float)
    parser.add_argument("--save_ver", help="Version number for the saved model. Default=1"), Default=1, type=int)
    args = parser.parse_args()
    if args.visualise_trg_example:
        # import libraries needed for visualisation
        visualisation_on()
    # call main function
    retrain_a_model(
        model_type=args.type,
        model_pretraining=args.pretraining,
        saving_name=args.saving_name,
        training_data=args.training_data,
        epochs=args.epochs,
        pg_sigma=args.sigma,
        on_hpc=args.cuda,
        show_trg_example=args.visualise_trg_example,
        learning_rate=args.learning_rate,
        save_version=args.save_ver,
    )
