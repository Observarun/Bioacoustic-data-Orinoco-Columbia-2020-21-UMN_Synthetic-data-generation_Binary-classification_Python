"""
It is recommended to execute this code in Colaboratory, else all the paths need to be changed. To use this file as is, Google Drive needs to be mounted.
"""

!pip install opensoundscape 

from opensoundscape.preprocess.preprocessors import CnnPreprocessor
from opensoundscape.torch.models.cnn import PytorchModel

import torch
import pandas as pd
from pathlib import Path
import numpy as np
import random
import subprocess
from matplotlib import pyplot as plt

import Preprocessing.py

# Setting seeds for python and pytorch
torch.manual_seed(17)
random.seed(17)


"""The following code cell is only meant to create an empty csv file w/ specified coloumns, at specified path, and need not be executed anymore."""

# Creating a dataframe for trained model's results on validation set.
validation_results_df = pd.DataFrame(columns=['# records_cattle pres (train+valid)', '# records_cattle abs (train+valid)', 'N_epochs', 'Batch size', 'Epoch_best model', 'F1 score_best model', 'Hardware acceleration', 'Training time'])
# Create csv file at the specified location using the above dataframe
validation_results_df.to_csv('/content/drive/Shareddrives/audio_data_cnn/Python programs_binary classification/validation_results.csv')

"""Training the model w/ training set and writing validation results to the csv file."""

validation_results_df = pd.read_csv('/content/drive/Shareddrives/audio_data_cnn/Python programs_binary classification/validation_results.csv', index_col=0)

epochs = 50
batch_size = 64
model = PytorchModel('resnet18', train_onehot_df.columns, single_target=True)

from datetime import datetime
start_time = datetime.now()

model.train(
    train_dataset=train_dataset,
    valid_dataset=valid_dataset,
    save_path='./binary_train/',
    epochs=epochs,
    batch_size=batch_size,
    save_interval=100,
    num_workers=0)
end_time = datetime.now()

import os
if int(os.environ["COLAB_GPU"]) > 0:
  hw_accel = 'GPU'
elif "COLAB_TPU_ADDR" in os.environ and os.environ["COLAB_TPU_ADDR"]:
  hw_accel = 'TPU'
else:
  hw_accel = 'None'

validation_results_df = validation_results_df.append({'# records_cattle pres (train+valid)': len(cattle_onehot_df[cattle_onehot_df['present']==1]), '# records_cattle abs (train+valid)': len(cattle_onehot_df[cattle_onehot_df['absent']==1]), 'N_epochs': epochs, 'Batch size': batch_size, 'Epoch_best model': model.best_epoch, 'F1 score_best model': model.best_f1, 'Hardware acceleration': hw_accel, 'Training time': end_time - start_time}, ignore_index=True)
validation_results_df.to_csv('/content/drive/Shareddrives/audio_data_cnn/Python programs_binary classification/validation_results.csv')

