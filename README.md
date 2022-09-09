# Binary Species Classification from Bio-acoustic Data


## Business Case

Human activity and climate change have been placing ever increasing pressure on biodiversity. Acoustic monitoring is seeing growing use in ecology and conservation biology research since it facilitates collection of data in a non-invasive manner continuously and over large areas. Such large scale data requires intensive analysis using advanced machine learning/deep learning algorithms.

In this work, which I have done in collaboration w/ a PhD student, Juliana Velez, in the Fieberg lab (Department of Fisheries, Wildlife, and Conservation Biology) at the University of Minnesota - Twin Cities, I have used a convolutional neural network (CNN) to classify audio files as having tapir sounds or not using [OpenSoundscape](https://github.com/kitzeslab/opensoundscape) library, which uses PyTorch under the hood.


## Table of Contents

1. [ Data ](#data)
2. [ Technologies ](#tex)
3. [ Executive Summary ](#exsum)


<a name="data"></a>
## Data

This work is part of a project for which the data is in the form of millions of audio & image files w/ various wildlife and disturbances in the form of domestic animals and gunshot sounds recorded in cattle ranches and forests of the Orinoquia region (also called the Eastern Plains) of Columbia. The broad goal of the project is to understand the interaction between wildlife and disturbances like cattle, dogs while also exploring the effect of poaching. For this work, we concern ourselves with mountain tapir (classified as "Endangered" by IUCN in 1996) and cattle.

A few thousand of the audio files have been categorised as having cattle sounds present or absent either mannually or using pattern recognition tools available online, like Rainforest Connection's Arbimon. The sub-folders cattle_pres and cattle_abs under Cattle folder contain these marked audio files. However, there are only a handful of audio records w/ tapir calls could be collected. The idea is to use this labelled audio data for training the model. Since CNNs work with image data, audio files are to be converted to frequency vs time spectrogram like images.

<a name="tex"></a>
## Technologies

Description...


<a name="exsum"></a>
## Executive Summary

The data files cattle_pres.csv and cattle_abs.csv contain names and other details (like date/time) of the audio files with cattle sounds present and absent respectively.
