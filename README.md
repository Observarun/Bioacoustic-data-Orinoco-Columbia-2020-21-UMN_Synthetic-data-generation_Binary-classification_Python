# Binary Species Classification from Bio-acoustic Data


## Business Case

Human activity and climate change have been placing ever increasing pressure on biodiversity. Acoustic monitoring is seeing growing use in ecology and conservation biology research since it facilitates collection of data in a non-invasive manner continuously and over large areas. Such large scale data requires intensive analysis using advanced machine learning or deep learning algorithms.

In this work, which I have done in collaboration w/ a PhD student, Juliana Velez, in the Fieberg lab (Department of Fisheries, Wildlife, and Conservation Biology) at the University of Minnesota - Twin Cities, I have used convolutional neural networks (CNNs) and Python programming to classify audio files as having an animal species sound present or absent using [OpenSoundscape](https://github.com/kitzeslab/opensoundscape) library, which uses PyTorch under the hood. I've also done data augmentation - generating new audio records from existing ones - using SciPy and PyDub libraries of Python, for a threatened species which there were few records for.


## Table of Contents

1. [ Data ](#data)
2. [ Technologies ](#tex)
3. [ Executive Summary ](#exsum)


<a name="data"></a>
## Data

This work is part of a project for which the data is in the form of millions of audio & image files w/ various wildlife and disturbances in the form of domestic animals and gunshot sounds recorded in the forests and adjoining cattle ranches in the Orinoquia region (also known as the Eastern Plains) of Columbia. The broad goal of the project is to understand the interaction between wildlife and domestic animals like cattle, dogs while also exploring the effect of poaching. For this repository, the species of concern are mountain tapir (classified as "Endangered" by IUCN in 1996) and cattle.

[AudioMoth](https://www.openacousticdevices.info/audiomoth) devices have been used to capture the sounds. Raw files are in FLAC (Free Lossless Audio Codec) or WAV format, each 10 seconds long in accordance with what has been set in the AudioMoth devices. They typically have forest noise in the background, like twigs snapping as the animals move around. A few thousand of the audio files have been categorised as having cattle sounds present or absent either mannually or using pattern recognition tools available online, like Rainforest Connection's Arbimon. These constitute the labeled data for cattle presence/absence problem. The sub-folders cattle_pres and cattle_abs under Cattle folder contain these marked audio files. The data files cattle_pres.csv and cattle_abs.csv contain names and other details (like date, time) of the audio files with cattle sounds present and absent respectively. The other species of interest is tapir. However, only four audio records w/ tapir calls could be obtained from the study site. Accordingly, some tapir data collected from Cali Zoo (Zoologico de Cali, Cali, Columbia) were also used.


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

For simplicity, the machine learning model in this repository is restricted to the binary classification problem. Accordingly, the model needs to be trained with a dataset containing records for the presence and absence of the species of concern (which is cattle). Additionally, the bigger project which this work is part of has tapir as one species of interest. Since only a handful of raw audio records w/ tapir sounds could be collected, I have performed data augmentation on them, so as to inflate the size of the tapir positive data. I ensured that all the audio clips resulting from this procedure are of 5 seconds duration. This would subsequently be useful in model training for tapir absence/presence (which is not part of this repository) because it is done sans hyperparameter tuning, and accordingly, clips of shorter duration should make the performance of a randomly chosen CNN model better. However, model training w/ too short clips is very slow, and 5 seconds offers a good balance.

As stated before, there are four audio files from the study site containing five tapir calls in all. I have understood from literature (**reference required**) that tapirs have more types of calls, and the five calls that I have got access to do not appear distinct. Accordingly, I requessted to include some of the tapir audio records obtained from the Cali Zoo. This increased the number of tapir files in the base dataset (which is used to generate extra recordings) to 18.

I have generated new data out of the original tapir recordings using two different approaches. To understand the mechanism of generating new data out of the existing tapir records, consider the representative spectrogram (which is a frequency vs time plot created from an audio clip) below.
    (Diagram)
The frequency spike as seen fairly localised in time represents a tapir call. I moved the tapir frequency band in time, with each new clip created out of this one having this frequency band at different time instances. I have done this in two different ways, distinction being in the background of the tapir call. Next, I describe and give arguments for each.

A model trained with this tapir data will need to isolate the tapir call out of the (say) 5 seconds long clip. Since the training data for tapir presence is sparse, it seems to make sense to have complete silence other than at the instance of the tapir call, which is typically less than a second. In a way, this makes it convenient for the model to isolate the tapir call out of the background. This is the approach I take in the first method - generate records having complete silence in the background of the tapir sound.

In the second approach, I generate new clips with real background - could be other animals/birds calls, leaves rustling, twigs snapping as creatures move around, et cetera. This approach makes sure that the subsequent model training is done with real data, like would be encountered during testing phase. As stated before, the duration of each raw clip is 10 seconds and the clips generated after this procedure are 5 seconds long. This brings the question what time duration (out of the 10 seconds) to choose the background from. One relatively simple way could be to use (or piece together, if required) a clip (at least 5 seconds long) containing only the 'background noises', and use a required section of that (length being $ 5 - length of tapir call $) together with a tapir sound at various instances to make new clips corresponding to each tapir call. However, the best answer in my opinion comes from the motivation of this (second) approach of generating new data. I wanted the model to see the real background (similar to what it would see in test conditions). Different test files would obviously have various types of noises in various combinations and orders. Hence, I chose to create audio clips w/ background taken from the 5 seconds chunk that the tapir call happens to be in. To exemplify, if the original 10 seconds audio has two tapir calls - one between 3, 4 sec and another between 6, 7 sec, then

Both these methods require the chunk of audio w/ tapir to be separated. This requires identifying the temporal location of the tapir sound in the clip. I played each record to identify, to within a second, the location of the tapir call, and looked for a discernable pattern within that one second in the corresponding spectrogram. To generate spectrograms from the acoustic file, an AudioSegment instance is created 

The idea is to use this labelled audio data for training the model. Since CNNs work with image data, audio files are to be converted to frequency vs time spectrogram like images. How spectrogram is created.
