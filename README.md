# Binary Species Classification from Bio-acoustic Data


## Business Case

Human activity and climate change have been placing ever increasing pressure on biodiversity. A common problem of interest in conservation biology and ecology research is to detect the presence of a wildlife species in a region. Acoustic monitoring, which is a technique that uses certain electronic devices to capture animal sounds, is seeing growing use in this context. It facilitates collection of wildlife data in a non-invasive manner continuously and over large areas, while avoiding the heavy cost associated with employing humanpower for manual surveys. However, the large scale data thus generated requires intensive analysis using advanced machine learning or deep learning algorithms.

I have done this work in collaboration w/ [Juliana Velez](github handle), a PhD student in the Fieberg Lab (PI: [John Fieberg](github handle), Department of Fisheries, Wildlife, and Conservation Biology) at the University of Minnesota - Twin Cities. I have used convolutional neural networks (CNNs) to classify audio files as having an animal species sound present or absent. Tapir is one of the wildlife species (classified as "Endangered" by IUCN in 1996) that Juliana has collected data for. Owing to insufficient data for tapir, I've also performed data augmentation - generating new audio records from existing ones. Though I do not train a model for detecting tapir in this repository, this will be done for subsequently, and new data will be useful for that.

All coding in this repository has been performed using Python. For writing CNN, I've used [OpenSoundscape](https://github.com/kitzeslab/opensoundscape) library, which uses pytorch under the hood. Data augmentation has been performed using SciPy and PyDub libraries. I have also written an autoencoder and a variational autoencoder for pre-processing/augmenting tapir data using PyTorch framework.


## Table of Contents

1. [ Data (Collection/description/set details/etymology!) ](#data)
2. [ Technologies ](#tex)
3. [ Executive Summary ](#exsum)


<a name="data"></a>
## Data

Acoustic data has been captured using [AudioMoth](https://www.openacousticdevices.info/audiomoth) devices. Each measures $58 \times 48 \times 15$, and is typically hung from a tree at a height of about 4 m. Most data has been recorded in the forests and adjoining cattle ranches in the Orinoquia region (also known as the Eastern Plains) of Columbia, in the period __.

There are millions of audio files, pertaining to wildlife and various disturbances in the form of domestic animals like cattle, dogs, and gunshot sounds. AudioMoth devices were configured to create 10 sec long audio clips in FLAC (Free Lossless Audio Codec) or WAV format. They typically have forest noise in the background, like other animals/birds calls, leaves rustling, twigs snapping as creatures move around, et cetera. A few thousand of these audio files have been labeled as having cattle sounds present or absent by Juliana, either manually or using pattern recognition tools available online (like Rainforest Connection's Arbimon). For the data augmentation task, as mentioned before, the species of concern is mountain tapir. Only four audio records w/ tapir calls could be obtained from the study site. Consequently, some tapir data collected from Cali Zoo (Zoologico de Cali, Cali, Columbia) were also used.

In this repository, I have summarised the data in CSV files. In addition, I provide links to the drive containing audio files. This, however, requires requesting access. The sub-directories cattle_pres and cattle_abs under Cattle directory contain the labeled audio files. Tapir data is under the corresponding directory.


<a name="tex"></a>
## Technologies

[Python 3](https://www.python.org/download/releases/3.0/)

[matplotlib](https://matplotlib.org/) library: Module matplotlib.pyplot for creating, displaying, and saving figures/colorplots.

[pathib](https://pathlib.readthedocs.io/en/pep428/) module: Classes Path and PurePath for working w/ files and directories.

[pandas](https://pandas.pydata.org/) package: For working w/ dataframes.

[Pydub](http://pydub.com/) library: Used in data augmentation. pydub.AudioSegment for extracting samples and metadata from a sound file, splitting an audio file into parts, and merging various files together.

[SciPy](https://scipy.org/) library: Module scipy.signal for creating spectrogram from an audio file.

[OpenSoundscape](http://opensoundscape.org/en/latest/) library: Used for training CNN model with bio-acoustic data.

[Pytorch](https://pytorch.org/) framework: Classes and sub-modules of torch.nn module used for writing autoencoder.


<a name="exsum"></a>
## Executive Summary (Work summary...)

The broad goal of the project is to understand the interaction between wildlife and domestic animals like cattle, dogs while also exploring the effect of poaching. The machine learning problem in this project pertains to identifying animal species in acoustic data obtained as explained earlier. Ideally, one would expect the model to identify all the species in the given data. However, for simplicity, the classification model in this repository is restricted to presence/absence of one species, which is a binary classification problem. For this purpose, I have focused on cattle. Accordingly, the model needs to be trained with a dataset containing records for the presence and absence of cattle. Additionally, the second problem addressed in this repository is pertains to tapir data. Since only a handful of raw audio records w/ tapir sounds could be collected, I have performed data augmentation on them, so as to inflate the size of the tapir positive dataset. This will potentially be useful for future work on tapir absence/presence classification problem.

### The insufficient data problem (Data augmentation for tapir)

As stated before, tapir data will be relevant for futur work involving ML modelling for identifying tapir presence/absence in the audio. Since there were four audio files from the study site containing five tapir calls in all, this was certainly insufficient to train the model. Further, the five tapir sounds I had were not all distinct. From literature reivew (**reference required**), I found that there are more types of tapir calls. While it's possible to generate new records using the existing ones, such data augmentation will not solve the representativeness problem just described. Accordingly, I requested the Fieberg Lab to include some of the tapir audio records obtained from the Cali Zoo.

Cali Zoo has one tapir enclosure (**check**) and the data is collected using three AudioMoth devices. Though these are placed at different locations, their range is long enough to capture tapir sounds from the enclosure. Each of them records and saves audio clips at various times of the day. The clips recorded by two or more AudioMoths at a particular time of the day have the same tapir sound, but with different background on account of their local neighbourhood. While this brings in much more tapir data, for the purpose of representativeness problem discussed in the last paragraph, the number of relevant new clips is just the 'union' of the clips from the 3 AudioMoths (each clip used exactly once). This increased the number of tapir files in the base dataset (which is used to generate extra recordings) to 18.

### Generating new data

Juliana was recommended by OpenSounscape developers to use 5 sec long clips for training the model written this framework. Accordingly, I ensured that all the audio clips resulting from the data augmentation procedure are of 5 seconds duration, so that they can be used for subsequent training of a model to detect tapir presence. In my opinion, the recommendation of 5 sec clips is in line with the way CNN model is trained for classification in the [OpenSoundscape tutorial](http://opensoundscape.org/en/latest/tutorials/cnn.html). As can be seen on this webpage, the machine learning classification task is performed sans hyperparameter tuning, which leaves making educated speculations about the parameters of the model. It is intuitively understood that clips of arbitrarily short duration should make it easier for the model to locate the tapir call amid the background, making the performance of a randomly chosen CNN model better. However, model training w/ too short clips is very slow, and 5 seconds offers a good balance.  

I have generated new data out of the original tapir recordings using two different approaches. To understand the mechanism of generating new data out of the existing tapir records, consider the spectrogram (which is a frequency vs time plot created from an audio clip) next.
    <p align="center">
    <img src="https://user-images.githubusercontent.com/83636458/194688632-39b7fe1f-4cf9-4c5b-8c48-1a63aa89d4a9.png">
    <em> Spectrogram of an audio clip containing one tapir call </em>
    </p>
The spike in frequency as seen fairly localised in time represents a tapir call. I moved the tapir frequency band in time, with each new clip created out of this one having this frequency band at different time instances. I have done this in two different ways, distinction being in the background of the tapir call. (Passive) Next, I describe and give arguments for each.

Bullet point) A model trained with this tapir data will need to isolate the tapir call out of the 5 seconds long clip. Since the training data for tapir presence is sparse, it seems to make sense to have complete silence other than at the instance of the tapir call, which is typically less than a second. In a way, this makes it convenient for the model to isolate the tapir call out of the background. This is the approach I take in the first method - generate records having complete silence in the background of the tapir sound. (Last line first.)

In the second approach, I generate new clips with real background - could be other animals/birds calls, leaves rustling, twigs snapping as creatures move around, et cetera. This approach makes sure that the subsequent model training is done with real data, like would be encountered during testing phase. As stated before, the duration of each raw clip is 10 seconds and the clips generated after this procedure are 5 seconds long. This brings the question what time duration (out of the 10 seconds) to choose the background from. One relatively simple way could be to use (or piece together, if required) a clip (at least 5 seconds long) containing only the 'background noises', and use a required section of that (length being $ 5 - length of tapir call $) together with various tapir sound at various instances to make new clips corresponding to each tapir call. However, the best answer in my opinion comes from the motivation of this (second) approach of generating new data. I wanted the model to see the real background (similar to what it would see in test conditions). Different test files would obviously have various types of noises in various combinations and orders. Hence, I chose to create audio clips w/ background taken from the 5 seconds chunk that the tapir call happens to be in. To exemplify, if the original 10 seconds audio has two tapir calls - one between 3, 4 sec and another between 6, 7 sec, then I use the first tapir sound and the first 5 seconds of the audio to generate clips, with the tapir sound at locations 0 sec, 1 sec, 2 sec, 3 sec, 4 sec. And using the tapir sound between 6, 7 sec, I use the section of the audio from 6 to 10 sec to create 5 clips.

### Spectrograms from audio

Both these methods require the chunk of audio w/ tapir sound to be separated, which requires identifying the temporal location of the tapir sound in the clip. I played (listened to) each record to identify, to within a second, the location of the tapir call, and looked for a discernable pattern within that one second in the corresponding spectrogram. Spectrogram is a frequency vs time representation of an acoustic signal generated using Fourier transformation. Essentially, it is a transformation from amplitude to frequency space.

I create spectrograms using scipy.signal's spectrogram() method, which takes the numpy array, created from Audiosegment object (pydub library) for the corresponding audio file, as an argument. An example can be seen above, such frequency blobs stacked on top of each other typically correspond to a nasal sound.

### Generating new clips (Code detail etc.)

As explained before, I generate new clips in two ways. For silence in background, I generate chunks of silence with silent() class method in pydub.AudioSegment. For each audio file, I create pydub.AudioSegment instance using AudioSegment.from_file('filename'), and extract a chunk of this file from a to b msec using AudioSegment.from_file('filename')[a:b] (**what is it called?!** https://github.com/jiaaro/pydub, https://stackoverflow.com/questions/42060433/python-pydub-splitting-an-audio-file). For saving the clips generated using a particular file, I use the stem property of pathlib.PurePath(filename) instance.

### Binary classification

These constitute the labeled data for cattle presence/absence problem. The idea is to use this labelled audio data for training the model. Since CNNs work with image data, audio files are to be converted to frequency vs time spectrogram like images. 
