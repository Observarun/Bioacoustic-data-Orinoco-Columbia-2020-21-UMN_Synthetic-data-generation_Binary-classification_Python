# Binary Species Classification and Augmentation from Bio-acoustic Data


## Business Case (or Motivation?)

Human activity and climate change have been placing ever increasing pressure on biodiversity. A common problem of interest in conservation biology and ecology research is to detect the presence of a wildlife species in a region. Acoustic monitoring, which is a technique that uses certain electronic devices to capture animal sounds, is seeing growing use in this context. It facilitates collection of wildlife data in a non-invasive manner continuously and over large areas, while avoiding the heavy cost associated with employing humanpower for manual surveys. However, the large scale data thus generated requires intensive analysis using advanced machine learning or deep learning algorithms.

I have done this work in collaboration w/ [Juliana Velez](github handle), a PhD student in the Fieberg Lab (PI: [John Fieberg](github handle), Department of Fisheries, Wildlife, and Conservation Biology) at the University of Minnesota - Twin Cities. I have used convolutional neural networks (CNNs) to classify audio files as having an cattle sounds present or absent. Tapir is one of the wildlife species (classified as "Endangered" by IUCN in 1996) that Juliana has collected data for. Owing to insufficient data for tapir, I've also performed data augmentation - generating new data using existing base dataset. Though I do not train a model for detecting tapir in this repository, this will be done subsequently, and generated data will be useful for that.

All coding in this repository has been performed using Python. For writing CNN, I've used [OpenSoundscape](https://github.com/kitzeslab/opensoundscape) framework, which comes as an open source library and uses PyTorch under the hood. Data augmentation has been performed using SciPy and PyDub libraries. I have also written a variational autoencoder, a state of the art deep learning algorithm, for pre-processing/augmenting tapir data using PyTorch framework.


## Table of Contents

1. [ Data ](#data)
2. [ Technologies ](#tex)
3. [ Executive Summary ](#exsum)


<a name="data"></a>
## Data (Collection/description/set details/etymology!)

Acoustic data has been captured using [AudioMoth](https://www.openacousticdevices.info/audiomoth) devices. Each measures $58 \times 48 \times 15$ mm, and is typically hung from a tree at a height of about 4 m. Most data has been recorded in the forests and adjoining cattle ranches in the Orinoquia region (also known as the Eastern Plains) of Columbia, in the period __(**check!**).

There are millions of audio files, pertaining to wildlife and various disturbances in the form of domestic animals like cattle, dogs, and gunshot sounds. AudioMoth devices were configured to create 10 sec long audio clips in FLAC (Free Lossless Audio Codec) or WAV format. They have forest noise in the background of one or more call by the animal species under consideration. The background could be other animals/birds calls, leaves rustling, twigs snapping as creatures move around, et cetera. A few thousand of these audio files have been labeled as having cattle sounds present or absent by Juliana, either manually or using pattern recognition tools available online (like Rainforest Connection's Arbimon). These are relevant for training the model to detect cattle absence/presence. For the data augmentation task, as mentioned before, the species of concern is mountain tapir. Only four audio records w/ tapir calls could be obtained from the study site. Consequently, some tapir data collected from Cali Zoo (Zoologico de Cali, Cali, Columbia) were also used.

In this repository, I have summarised the cattle data in CSV files. In addition, I provide links to the drive containing audio clips and spectrogram images. This, however, requires requesting access. The sub-directories cattle_pres and cattle_abs under Cattle directory contain the labeled audio files. Tapir data is under the corresponding directory.


<a name="tex"></a>
## Technologies

[Python 3](https://www.python.org/download/releases/3.0/)

[matplotlib](https://matplotlib.org/) library: Module matplotlib.pyplot for creating, displaying, and saving figures/colorplots.

[pathib](https://pathlib.readthedocs.io/en/pep428/) module: Classes Path and PurePath for working w/ files and directories.

[pandas](https://pandas.pydata.org/) package: For working w/ dataframes.

[Pydub](http://pydub.com/) library: Used in data augmentation. pydub.AudioSegment for extracting samples and metadata from a sound file, splitting an audio file into parts, and merging various files together.

[SciPy](https://scipy.org/) library: Module scipy.signal for creating spectrogram from an audio file.

[OpenSoundscape](http://opensoundscape.org/en/latest/) framework: Corresponding library used for coding CNN models to analyse bio-acoustic data.

[Pytorch](https://pytorch.org/) framework: Classes and sub-modules of torch.nn used for writing variational autoencoder.


<a name="exsum"></a>
## Executive Summary (or Work summary...)

The broad goal of the project is to understand the interaction between wildlife and domestic animals like cattle, dogs while also exploring the effect of poaching through gunshots. The machine learning problem in this project pertains to identifying animal species in acoustic data collected as explained earlier. A common approach to working with sounds (e.g., in speech recognition) is to generate a spectrogram, which is a diagram showing how frequency varies with time (i.e., a frequency vs time plot) in an acoustic signal. Essentially, it is a Fourier transformation from amplitude to frequency space. For the problems in this repository, the goal is not to identify a sequence of letters from a sound (which is so in speech recognition), but spectrograms offer a very useful starting point.

Ideally, one would expect the model to identify all the species in the given data. However, for simplicity, the classification model in this repository is restricted to presence/absence of one species, which is a binary classification problem. For this purpose, I have focused on cattle. Accordingly, the model needs to be trained with a dataset containing records for the presence and absence of cattle. Here, a 2-dimensional convolutional network is used for this purpose, which performs best on data with spatial features. In an image, nearby pixels are usually related to each other thereby making spectrograms relevant for use w/ a CNN. Additionally, the second problem addressed in this repository pertains to tapir data. Since only a handful of raw audio records w/ tapir sounds could be collected, data augmentation needs to be performed on them, so as to inflate the size of the tapir positive/presence dataset. I have done this using two methods, at the level of audio signals and at the level of spectrograms. This will potentially be useful for future work on tapir absence/presence classification modelling problem.

### Problems with sparse tapir data

As stated before, tapir data will be relevant for future work involving ML modelling for identifying tapir presence/absence in the audio. Since there were four audio files from the study site containing five tapir calls in all, this was certainly insufficient to train the model. Further, the five tapir sounds I had were not all distinct. From [literature](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8065771/), I found that there are more types of tapir calls. While it's possible to generate new records using the existing ones, such data augmentation will not solve the representativeness problem just described. Accordingly, I requested the Fieberg Lab to include some of the tapir audio records obtained from the Cali Zoo.

Cali Zoo has one tapir enclosure (**confirm!**) and the data is collected using three AudioMoth devices. Though these are placed at different locations, their range is long enough to capture tapir sounds from the tapir enclosure. Each of them records and saves audio clips at various times of the day. The clips recorded by two or more AudioMoths at a particular time of the day have the same tapir sound, but with different background on account of the location of each AudioMoth device. While this brings in much more tapir data, for the purpose of representativeness problem discussed in the last paragraph, the number of relevant new clips is just the 'union' of the clips from the 3 AudioMoths (each clip used exactly once). This increased the number of tapir files in the base dataset (which is used to generate new data) to 10.

### Generating new audio clips

Juliana was recommended by OpenSounscape developers to use 5 sec long clips for training the model written using this framework. Accordingly, I ensured that all the audio clips resulting from the data augmentation procedure are of 5 seconds duration, so that they can be used for subsequent training of a model to detect tapir presence. In my opinion, the recommendation of 5 sec clips is in line with the way convolutional network is trained for classification in the OpenSoundscape's [CNN tutorial](http://opensoundscape.org/en/latest/tutorials/cnn.html). As can be seen on this webpage, the machine learning classification task is performed sans hyperparameter tuning, which leaves making educated speculations about certain parameters of the model (e.g., batch size). (At this point, it is worth mentioning that OpSo does have a default implementation of learning rate scheduling in the form of stair case delay. Learning rate is multiplied by a cooling factor of $.7$ every $10$ epochs. The [advanced tutorial](http://opensoundscape.org/en/latest/tutorials/cnn_training_advanced.html) also shows a way to modify the scheduling by changing the cooling factor.) It is intuitively understood that clips of arbitrarily short duration should make it easier for the model to locate the tapir call amid the background, making the performance of a randomly chosen CNN model better. However, model training w/ too short clips is very slow, and 5 seconds offers a good balance.

The first method of data augmentation is what I would like to call 'controlled' augmentation. (The choice of this terminology will be clear soon.) Within this method, I have generated new data out of the original tapir recordings using two different approaches. To understand the mechanism, consider the spectrogram (which is a frequency vs time plot created from one of the audio clips) shown next.
    <p align="center">
    <img src="https://user-images.githubusercontent.com/83636458/194688632-39b7fe1f-4cf9-4c5b-8c48-1a63aa89d4a9.png"/>
        <br>
        <em> Spectrogram of an audio clip containing one tapir call </em>
    </p>
The spike in frequency as seen fairly localised between 7, 8 sec on time axis represents a tapir call. I moved the tapir frequency spike in time, with each new clip created out of this one having the frequency spike at different time instances. I have done this in two different ways, distinction being in the background of the tapir call. Next, I describe, and give arguments for, each.

* In the first approach, I generate records having complete silence in the background of the tapir sound. To understand the rationale for this, consider a hypothetical model trained with audio clips, each having at least one tapir sound. During the training phase, the model will need to isolate the tapir call out of the 5 seconds long clip. Since the training data for tapir presence is sparse, it seems to make sense to have complete silence other than at the instance of the tapir call. In a way, this makes it convenient for the model to isolate the tapir call out of the background.

* In the second approach, I use the guiding principle that the subsequent model for detecting tapir absence/presence should be trained with real data, like would be encountered during the testing phase. Different test files would obviously have various types of background noises in various combinations and orders. Accordingly, I generate new 5 sec clips containing a tapir sound and forest noises in the background. One way to make this project more relevant to the larger community is to use background noises from different landscapes, in the background of tapir calls. However, given the small amount of base data, I have decided to stick with the backgrounds relevant to this project, which, as mentioned before, could be other animals/birds calls, leaves rustling, twigs snapping as creatures move around, et cetera. This still leaves the following question: how to choose the background forest noises of duration $(5 - length$ $of$ $tapir$ $call)$ seconds from the 10 sec long clip? There are indeed various ways, the best and the simplest one, in my opinion, is the following. Use the background from either the first 5 sec or the last 5 sec chunk of the 10 sec raw clip, depending on which one the tapir call happens to be in. To exemplify, if the original 10 seconds audio clip has two tapir calls - one between 3, 4 sec and another between 6, 7 sec, then I use the first tapir sound and the first 5 seconds of the audio to generate 5 clips, with the tapir sound starting at locations 0 sec, 1 sec, 2 sec, 3 sec, 4 sec. And using the tapir sound between 6, 7 sec in the raw clip, I use the section of the audio from 6 to 10 sec to create 5 clips.

Both these methods require the chunk of audio w/ tapir sound to be separated out of the 10 sec long audio signal, which requires identifying the temporal location of the tapir sound in the clip. I played (listened to) each record to identify, to within a second, the location of the tapir call, and looked for a discernable pattern within that one second in the corresponding spectrogram.

#### Code specifics

As explained before, I have generated synthetic audio clips in two ways. For silence in background, I generated chunks of silence with silent() class method of AudioSegment. For each audio file, I created an AudioSegment instance using from_file('filename') class method, and extracted a chunk of this clip from a to b msec using [a:b] (similar to slicing a NumPy array). For naming new clips generated using a particular clip, I used the stem property of PurePath('existing_clip_name') instance.

I created spectrograms using spectrogram() method of scipy.signal module. This method takes a numpy array, created from Audiosegment object for the corresponding audio file, as an argument. An example can be seen in the figure above. Such frequency blobs stacked on top of each other typically correspond to a nasal sound from a mammal.

### An alternative - controlled spectrogram augmentation

At this point, it is worth noting that in this method of data augmentation, I've worked at the level of audio clips. It is equally possible to enhance the dataset using spectrograms corresponding to the audio clips in the base dataset. Since the spectrogram is a pictorial frequency vs time representation, this involves working with images. One advantage of working with spectrograms in a controlled manner is that both frequency band and time duration of tapir sound are accessible. Such an analysis has been done in [literature](https://arxiv.org/abs/1904.08779), and exploits both frequency-masking and time-masking. In this repository, I choose to not perform controlled augmentation of spectrogram data since the analysis goes on similar lines as audio augmentation. However, in the next section, I demonstrate the use of a state of the art deep learning algorithm for enhancing spectrogram data. This could be useful for those in the larger community who want to do modelling starting from spectrogram images.

### Variational autoencoder to generate synthetic spectrogram images

Variational autoencoder is a neural network based generative (in that it attempts to identify the structure of the data so as to simulate the data generation process), unsupervised (in that it doesn't require class labels for training) algorithm. It was proposed by [Kingma and Welling in 2013](https://doi.org/10.48550/arXiv.1312.6114). Consider a neural network that applies a set of non-linear transformations to the input data (to reduce its dimension) and maps it to a probability distribution, from which a latent vector is sampled. This network is an encoder. Another neural network, a decoder, then maps the latent vector back to the original input space using non-linear transformations. Essentially, the encoder compresses the data while the decoder decompresses it.
    <p align="center">
    <img src="https://user-images.githubusercontent.com/83636458/201509447-10f583a6-3b2f-46cc-97c8-60e47633eba6.png"/>
        <br>
        <em> Representation of a variational autoencoder. Image sourced from [internet](https://avandekleut.github.io/vae/) </em>
    </p>
As stated before, the input data is encoded as distribution over latent space random variable. The latent space thus obtained is continuous (as opposed to vae's architectural cousin - an autoencoder - where input data is deterministically mapped to the latent space). Further, one of the loss functions to be minimised during training is Kulback-Leibler divergence, which also penalises the distribution for deviating from standard Gaussian. This effectively enforces the requirement that the variance should be close to an identity matrix and mean should be small. This makes the generative process possible.

Let's call the latent random variable $z$. Suppose the encoded distribution is $q_\phi(z|x)$, where $x$ is the training data. Hence, $q\phi(z|x)$ corresponds to the encoder. Decoder is the likelihood $p_\theta(x|z)$, where $\theta$ represents the parameters of the model. $q_\phi(z|x)$ is the variational approximation to the posterior of the model, and accordingly, $\phi$ represents the variational parameters. Posterior distribution is initially represented by a prior $p(z)$ (which is assumed to be a unit normal distribution N(0,1)), that will be subsequently updated. As mentioned earlier, the latent random variable is sampled from the encoded distribution, i.e. $z ~ q_\phi(z|x)$. This makes the gradient calculation one of the terms in KL divergence (which is required during backpropagation) cumbersome. To overcome this, the authors proposed a reparametrisation trick, which involves expressing the random variable $z$ as a deterministic variable. If it is assumed that the posterior is approximately normal, then so would the variational approximation to the posterior ($q_\phi(z|x)$) be. In this case, the reparametrised latent vector can be represented as $z = \mu + \sigma\epsilon$, where $\epsilon$ is an auxillary noise variable $\epsilon \tilde N(0,1)$.

### Binary classification

Labeled data for cattle presence/absence problem. The idea is to use this labelled audio data for training the model.

Since CNNs work with image data, audio files are to be converted to frequency vs time spectrogram like images.

Tune learning rate, batch size. To make this model relevant for tapir data, tune # epochs too since don't have a stopping criterion (no validation set).

All clips same duration so that spectrograms easily fit into GPU memory.

1D ConvNets (employing time-only convolutions) have been used directly on acoustic waveform (). However, their performance is subdued (https://arxiv.org/abs/1610.00087) since acoustic signal is a sequence and CNNs don't have the ability to retain information from prior inputs. A CNN-RNN (i.e., a recurrent layer at the end of a convolutional network) has been shown to produce better results. [This article](https://peerj.com/articles/13152.pdf) explains this issues well.
