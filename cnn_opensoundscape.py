# -*- coding: utf-8 -*-

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

# Setting seeds for python and pytorch
torch.manual_seed(17)
random.seed(17)

"""To begin w/, need to create a dataframe for the data in both the csv files with one-hot encoding for presence and absence of cattle. """

# Create dataframe from cattle_pres.csv
cattle_pres_df = pd.read_csv('/content/drive/Shareddrives/audio_data_cnn/cattle_pres.csv', usecols = ['filename'], nrows=None)
# Replace all '/' w/ '_' so that the records under 'filename' are same as names of audio files
cattle_pres_df['filename'] = cattle_pres_df['filename'].str.replace('/','_')
# File names should carry complete path of the audio files so that they can be downloaded (and converted to spectrograms, etc.)
cattle_pres_df.filename = ['/content/drive/Shareddrives/audio_data_cnn/cattle_pres/'+f for f in cattle_pres_df.filename]
# Insert features, 'present' and 'absent', and fill them w/ 1 & 0 respectively
cattle_pres_df.insert(1, 'present', 1)
cattle_pres_df.insert(1, 'absent', 0)

# Repeat the above procedure starting w/ cattle_abs.csv
cattle_abs_df = pd.read_csv('/content/drive/Shareddrives/audio_data_cnn/cattle_abs.csv', usecols = ['filename'], nrows=None)
cattle_abs_df['filename'] = cattle_abs_df['filename'].str.replace('/','_')
cattle_abs_df.filename = ['/content/drive/Shareddrives/audio_data_cnn/cattle_abs/'+f for f in cattle_abs_df.filename]
cattle_abs_df.insert(1, 'present', 0)
cattle_abs_df.insert(1, 'absent', 1)

# Concatenate the two dataframes
cattle_onehot_df = pd.concat([cattle_pres_df, cattle_abs_df])
# Convert filename feature to index (as suggested in the tutorial). The dataframe so obtained is in 'one-hot form' just like in the tutorial.
cattle_onehot_df = cattle_onehot_df.set_index('filename')

"""Alternatively, can create one-hot vector using OneHotEncoder from sklearn."""

# Create dataframes for cattle presence. Insert a feature 'cattle' in the dataframe and populate w/ 'present'
cattle_pres_df = pd.read_csv('/content/drive/Shareddrives/audio_data_cnn/cattle_pres.csv', usecols = ['filename'])
cattle_pres_df['filename'] = cattle_pres_df['filename'].str.replace('/','_')
cattle_pres_df.filename = ['/content/drive/Shareddrives/audio_data_cnn/cattle_pres/'+f for f in cattle_pres_df.filename]
cattle_pres_df.insert(1, 'cattle', 'present')

# Repeat for cattle absence
cattle_abs_df = pd.read_csv('/content/drive/Shareddrives/audio_data_cnn/cattle_abs.csv', usecols = ['filename'])
cattle_abs_df['filename'] = cattle_abs_df['filename'].str.replace('/','_')
cattle_abs_df.filename = ['/content/drive/Shareddrives/audio_data_cnn/cattle_abs/'+f for f in cattle_abs_df.filename]
cattle_abs_df.insert(1, 'cattle', 'absent')

# Merge the two dataframes created above
cattle_df = pd.concat([cattle_pres_df, cattle_abs_df], ignore_index=True)

# One-hot encoding

from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()  # Instantiating OneHotEncoder

# Use 'cattle' feature to create labels
# Step 1: enc.fit_transform(cattle_df[['cattle']])
# Step 2: enc.fit_transform(cattle_df[['cattle']]).toarray()
# Step 3: pd.DataFrame(enc.fit_transform(cattle_df[['cattle']]).toarray())
# Step 4:
cattle_onehot_df = cattle_df.join(pd.DataFrame(enc.fit_transform(cattle_df[['cattle']]).toarray()))
# Drop 'cattle' and rename the labels
cattle_onehot_df.drop('cattle', inplace=True, axis=1)
cattle_onehot_df.rename(columns={0:'absent', 1:'present'}, inplace=True)

cattle_onehot_df = cattle_onehot_df.set_index('filename')

"""Preprocessing data."""

# Split the data into two sets - for training and validation - using train_test_split() method from sklearn
from sklearn.model_selection import train_test_split
train_onehot_df, valid_onehot_df = train_test_split(cattle_onehot_df, test_size=.2, random_state=1)

# Instantiate CnnPreprocessor and pass the dataframe to its constructor. This preprocesses the data and returns a pytorch tensor.
train_dataset = CnnPreprocessor(train_onehot_df)
valid_dataset = CnnPreprocessor(valid_onehot_df)

"""Inspect training images like in the tutorial."""

# show_tensor() method from the tutorial
def show_tensor(sample):
    plt.imshow((sample['X'][0,:,:]/2+0.5)*-1,cmap='Greys',vmin=-1,vmax=0)
    plt.show()

for i, d in enumerate(train_dataset.sample(n=4)):
    print(f"cattle_onehot_df: {d['y']}")
    show_tensor(d)

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
