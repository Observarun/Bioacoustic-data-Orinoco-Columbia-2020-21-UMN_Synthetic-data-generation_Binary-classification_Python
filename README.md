# Binary Species Classification from Bio-acoustic Data


## Business Case

Human activity and climate change have been placing ever increasing pressure on biodiversity. Acoustic monitoring is seeing growing use in ecology and conservation biology research since it facilitates collection of data in a non-invasive manner continuously and over large areas. Such large scale data requires intensive analysis using advanced machine learning or deep learning algorithms.

In this work, which I have done in collaboration w/ a PhD student, Juliana Velez, in the Fieberg lab (Department of Fisheries, Wildlife, and Conservation Biology) at the University of Minnesota - Twin Cities, I have used convolutional neural networks (CNNs) and Python programming to classify audio files as having an animal species sound present or absent using [OpenSoundscape](https://github.com/kitzeslab/opensoundscape) library, which uses PyTorch under the hood. I've also done data augmentation - generated extra audio records from existing ones - using SciPy and PyDub libraries of Python, for a threatened species which there were few records for.


## Table of Contents

1. [ Data ](#data)
2. [ Technologies ](#tex)
3. [ Executive Summary ](#exsum)


<a name="data"></a>
## Data

This work is part of a project for which the data is in the form of millions of audio & image files w/ various wildlife and disturbances in the form of domestic animals and gunshot sounds recorded in the forests and adjoining cattle ranches in the Orinoquia region (also called the Eastern Plains) of Columbia. The broad goal of the project is to understand the interaction between wildlife and domestic animals like cattle, dogs while also exploring the effect of poaching. For this repository, the species of concern are mountain tapir (classified as "Endangered" by IUCN in 1996) and cattle.

[AudioMoth](https://www.openacousticdevices.info/audiomoth) devices have been used to capture the sounds. Raw files are in FLAC (Free Lossless Audio Codec) or WAV format, each 10 seconds long in accordance with what has been set in the AudioMoth devices. They typically have forest noise in the background, like twigs snapping as the animals move around. A few thousand of the audio files have been categorised as having cattle sounds present or absent either mannually or using pattern recognition tools available online, like Rainforest Connection's Arbimon, which constitutes labeled data for cattle presence/absence problem. The sub-folders cattle_pres and cattle_abs under Cattle folder contain these marked audio files. The data files cattle_pres.csv and cattle_abs.csv contain names and other details (like date/time) of the audio files with cattle sounds present and absent respectively. The other species of interest is tapir. However, only four audio records w/ tapir calls could be obtained from the study site. The idea is to use this labelled audio data for training the model. Since CNNs work with image data, audio files are to be converted to frequency vs time spectrogram like images.

<a name="tex"></a>
## Technologies

Python

Matplotlib: Module matplotlib.pyplot for creating, displaying, and saving figures/colorplots.

Pathlib: Class pathlib.Path for extracting files out of and exporting/saving files to a drive location.

Pandas: For working w/ dataframes.

Pydub: Used in data augmentation. pydub.AudioSegment for extracting samples and metadata from a sound file, splitting an audio file into parts, and merging various files together.

Scipy: Module scipy.signal for creating spectrogram from an audio file.



<a name="exsum"></a>
## Executive Summary

The raw audio files are in FLAC (Free Lossless Audio Codec) format, and typically have forest noise in the background, like twigs snapping as the animals move around. For simplicity, this repository is restricted to the binary classification problem. Accordingly, the model needs to be trained with a dataset containing records for both the presence and absence of the species of concern (which is cattle). Additionally, the bigger project which this work is part of has tapir as one species of interest. Since only 4 raw audio records w/ tapir sounds could be collected, data augmentation needs to be performed on them so as to inflate the size of the tapir positive data, which would be useful to train the model with tapir calls.

The new records have been generated out of the existing fourFurther, as the model has fewer records (on a/c of sparse training data) to pick the tapir sound out of the clip, it was deemed important to train the model with each record having complete silence other than at the instance (typically less than a second) of tapir call. 

To prepare this kind of clip, the chunk of audio w/ tapir call needs to be separated. This requires identifying the temporal location of the tapir sound in the clip. I played each record to identify, up to a second, the location of the tapir call, and looked for a discernable pattern within that one second in the corresponding spectrogram. To generate spectrograms from the acoustic file, an AudioSegment instance is created 
