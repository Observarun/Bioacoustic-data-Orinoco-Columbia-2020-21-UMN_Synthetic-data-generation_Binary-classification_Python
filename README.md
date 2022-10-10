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

[AudioMoth](https://www.openacousticdevices.info/audiomoth) devices have been used to capture the sounds. Raw files are in FLAC (Free Lossless Audio Codec) or WAV format, each 10 seconds long in accordance with what has been set in the AudioMoth devices. They typically have forest noise in the background, like twigs snapping as the animals move around. A few thousand of the audio files have been categorised as having cattle sounds present or absent either mannually or using pattern recognition tools available online, like Rainforest Connection's Arbimon. The sub-folders cattle_pres and cattle_abs under Cattle folder contain these marked audio files. The data files cattle_pres.csv and cattle_abs.csv contain names and other details (like date, time) of the audio files with cattle sounds present and absent respectively. The other species of interest is tapir. However, only four audio records w/ tapir calls could be obtained from the study site. Accordingly, some tapir data collected from Cali Zoo (Zoologico de Cali, Cali, Columbia) were also used.


<a name="tex"></a>
## Technologies

Python

matplotlib: Module matplotlib.pyplot for creating, displaying, and saving figures/colorplots.

pathib: Class pathlib.Path for extracting files out of and exporting/saving files to a drive location.

Pandas: For working w/ dataframes.

Pydub: Used in data augmentation. pydub.AudioSegment for extracting samples and metadata from a sound file, splitting an audio file into parts, and merging various files together.

Scipy: Module scipy.signal for creating spectrogram from an audio file.


<a name="exsum"></a>
## Executive Summary

For simplicity, the machine learning model in this repository is restricted to the binary classification problem. Accordingly, the model needs to be trained with a dataset containing records for the presence and absence of the species of concern (which is cattle). Additionally, the bigger project which this work is part of has tapir as one species of interest. Since only a handful of raw audio records w/ tapir sounds could be collected, I have performed data augmentation on them, so as to inflate the size of the tapir positive data. This will subsequently be useful for the tapir absence/presence classification problem.

### The insufficient data problem

As stated before, there are four audio files from the study site containing five tapir calls in all. This is insufficient not just to train the model, but also at another level. I have understood from literature (**reference required**) that tapirs have more types of calls, and the five calls that I have got access to do not appear distinct. Accordingly, I requested the Fieberg Lab to include some of the tapir audio records obtained from the Cali Zoo. This zoo data is collected using three AudioMoth devices, and has overlapping clips w/ same tapir sounds recorded by them but w/ somewhat different background (on a/c of their locations in the zoo). While this brings in much more tapir data, for the purpose of representativeness problem, the number of relevant new clips is just the 'union' of the clips from the 3 AudioMoths (each clip used exactly once). This increased the number of tapir files in the base dataset (which is used to generate extra recordings) to 18.

### Generating new data

I ensured that all the audio clips resulting from this procedure are of 5 seconds duration. This is in line with the way the data science problem problem in the bigger project, which the work in this repository is part of, has been addressed by the PhD student. The machine learning classification task is performed sans hyperparameter tuning, which leaves making educated speculations about the parameters of the model. Clips of shorter duration should make it easier for the model to locate the tapir call amid the background, making the performance of a randomly chosen CNN model better. However, model training w/ too short clips is very slow, and 5 seconds offers a good balance.

I have generated new data out of the original tapir recordings using two different approaches. To understand the mechanism of generating new data out of the existing tapir records, consider the spectrogram (which is a frequency vs time plot created from an audio clip) next.
    ![Spectrogram corresponding an audio clip](https://user-images.githubusercontent.com/83636458/194688632-39b7fe1f-4cf9-4c5b-8c48-1a63aa89d4a9.png)
The spike in frequency as seen fairly localised in time represents a tapir call. I moved the tapir frequency band in time, with each new clip created out of this one having this frequency band at different time instances. I have done this in two different ways, distinction being in the background of the tapir call. Next, I describe and give arguments for each.

A model trained with this tapir data will need to isolate the tapir call out of the 5 seconds long clip. Since the training data for tapir presence is sparse, it seems to make sense to have complete silence other than at the instance of the tapir call, which is typically less than a second. In a way, this makes it convenient for the model to isolate the tapir call out of the background. This is the approach I take in the first method - generate records having complete silence in the background of the tapir sound.

In the second approach, I generate new clips with real background - could be other animals/birds calls, leaves rustling, twigs snapping as creatures move around, et cetera. This approach makes sure that the subsequent model training is done with real data, like would be encountered during testing phase. As stated before, the duration of each raw clip is 10 seconds and the clips generated after this procedure are 5 seconds long. This brings the question what time duration (out of the 10 seconds) to choose the background from. One relatively simple way could be to use (or piece together, if required) a clip (at least 5 seconds long) containing only the 'background noises', and use a required section of that (length being $ 5 - length of tapir call $) together with various tapir sound at various instances to make new clips corresponding to each tapir call. However, the best answer in my opinion comes from the motivation of this (second) approach of generating new data. I wanted the model to see the real background (similar to what it would see in test conditions). Different test files would obviously have various types of noises in various combinations and orders. Hence, I chose to create audio clips w/ background taken from the 5 seconds chunk that the tapir call happens to be in. To exemplify, if the original 10 seconds audio has two tapir calls - one between 3, 4 sec and another between 6, 7 sec, then I use the first tapir sound and the first 5 seconds of the audio to generate clips, with the tapir sound at locations 0 sec, 1 sec, 2 sec, 3 sec, 4 sec. And using the tapir sound between 6, 7 sec, I use the section of the audio from 6 to 10 sec to create 5 clips.

### Spectrograms from audio

Both these methods require the chunk of audio w/ tapir sound to be separated, which requires identifying the temporal location of the tapir sound in the clip. I played each record to identify, to within a second, the location of the tapir call, and looked for a discernable pattern within that one second in the corresponding spectrogram. Spectrogram is a frequency vs time representation of an acoustic signal generated using Fourier transformation. Essentially, it is a transformation from amplitude to frequency space.

Further, for the classification problem addressed in this repository, convolutional neural networks are used. Since CNNs work w/ image data, spectrograms corresponding to all the audio files need to be generated. I create spectrograms using scipy.signal's spectrogram() method, which takes the numpy array, created from Audiosegment object (pydub library) for the corresponding audio file, as an argument. An example can be seen above, such frequency blobs stacked on top of each other typically correspond to a nasal sound.

### Generating new clips

As explained before, I generate new clips in two ways. For silence in background, I generate chunks of silence with silent() class method in pydub.AudioSegment. For each audio file, I create pydub.AudioSegment instance using AudioSegment.from_file('filename'), and extract a chunk of this file from a to b msec using AudioSegment.from_file('filename')[a:b] (**what is it called?!** https://github.com/jiaaro/pydub, https://stackoverflow.com/questions/42060433/python-pydub-splitting-an-audio-file). For saving the clips generated using a particular file, I use the stem property of pathlib.PurePath(filename) instance.

### Binary classification

These constitute the labeled data for cattle presence/absence problem. The idea is to use this labelled audio data for training the model. Since CNNs work with image data, audio files are to be converted to frequency vs time spectrogram like images. How spectrogram is created.
