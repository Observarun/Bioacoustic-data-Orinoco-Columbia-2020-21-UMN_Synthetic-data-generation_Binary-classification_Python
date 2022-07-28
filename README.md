# Binary Species Classification from Bio-acoustic Data


## Business Case

Human activity and climate change have been placing ever increasing pressure on biodiversity. Acoustic monitoring is seeing growing use in ecology and conservation biology research since it facilitates collection of data in a non-invasive manner continuously and over large areas. Such large scale data requires intensive analysis using advanced machine learning/deep learning algorithms.

In this work, which I have done in collaboration w/ a PhD student, Juliana Velez, in the Fieberg lab (Department of Fisheries, Wildlife, and Conservation Biology) at the University of Minnesota - Twin Cities, I have used a convolutional neural network (CNN) to classify audio files as having cattle sounds or not using [OpenSoundscape](https://github.com/kitzeslab/opensoundscape) library, which uses PyTorch under the hood.


## Table of Contents

1. [ Data ](#data)
2. [ Technologies ](#tex)
3. [ Executive Summary ](#exsum)


<a name="data"></a>
## Data

Data collected for this project is in the form of millions of audio files w/ animal sounds recorded in fields and forests. A few thousand of these files have been categorised as having cattle sounds present or absent either mannually or using pattern recognition tools available online, like Rainforest Connection's Arbimon.

The folders cattle_pres and cattle_abs contain these marked audio files. The idea is to use this labelled audio data for training the model. Since CNNs work with image data, audio files are to be converted to frequency vs time spectrogram like images. The data files cattle_pres.csv and cattle_abs.csv contain names and other details (like date/time) of the audio files with cattle sounds present and absent respectively.


<a name="tex"></a>
## Technologies

Description...


<a name="exsum"></a>
## Executive Summary

Description...
