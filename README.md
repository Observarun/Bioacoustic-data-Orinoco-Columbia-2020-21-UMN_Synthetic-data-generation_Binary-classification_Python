# Binary Species Classification and Augmentation using Bio-acoustic Data


<!--## Business Case-->
## Motivation

Human activity and climate change have been placing ever increasing pressure on biodiversity. A common problem of interest in conservation biology and ecology research is to detect the presence of a wildlife species in a region. Acoustic monitoring, which is a technique that uses certain electronic devices to capture animal sounds, is seeing growing use in this context. It facilitates collection of wildlife data in a non-invasive manner continuously and over large areas, while avoiding the heavy cost associated with employing humanpower for manual surveys. However, the large scale data thus generated requires intensive analysis using advanced machine learning or deep learning algorithms.

I have done this work in collaboration w/ [Juliana Velez](github handle), a PhD student in the Fieberg Lab (PI: [John Fieberg](github handle), Department of Fisheries, Wildlife, and Conservation Biology) at the University of Minnesota - Twin Cities. I have used convolutional neural networks (CNNs) to classify audio files as having an cattle sounds present or absent. Tapir is one of the wildlife species (classified as "Endangered" by IUCN in 1996) that Juliana has collected data for. Owing to insufficient data for tapir, I've also performed data augmentation - generating new data using existing base dataset. Though I do not train a model for detecting tapir in this repository, this will be done subsequently, and generated data will be useful for that.

All coding in this repository has been performed using Python. <!--For writing CNN, I've used [OpenSoundscape](https://github.com/kitzeslab/opensoundscape) framework, which comes as an open source library and uses PyTorch under the hood.--> Data augmentation has been performed using SciPy and PyDub libraries. I have also written a variational autoencoder, a state of the art deep learning algorithm, for pre-processing/augmenting tapir data using PyTorch framework.


## Table of Contents

1. [ Data ](#data)
2. [ Technologies ](#tex)
3. [ Introduction ](#intro)
4. [ Spectrogram ](#spec)
5. [ Code specifics ](#speccode)



<a name="data"></a>
<!--## Data (Collection/description/set details/etymology!)-->
## Data

Acoustic data has been captured using [AudioMoth](https://www.openacousticdevices.info/audiomoth) devices. Each measures $58 \times 48 \times 15$ mm, and is typically hung from a tree at a height of about 4 m. Most data has been recorded in the forests and adjoining cattle ranches in the Orinoquia region (also known as the Eastern Plains) of Columbia, in the period 2020-21.

There are millions of audio files, pertaining to wildlife and various disturbances in the form of domestic animals like cattle, dogs, and gunshot sounds. AudioMoth devices were configured to create 10 sec long audio clips in FLAC (Free Lossless Audio Codec) or WAV format. They have forest noise in the background of one or more call by the animal species under consideration. The background could be other animals/birds calls, leaves rustling, twigs snapping as creatures move around, et cetera. A few thousand of these audio files have been labeled as having cattle sounds present or absent by Juliana, either manually or using pattern recognition tools available online (like Rainforest Connection's Arbimon). These are relevant for training the model to detect cattle absence/presence. For the data augmentation task, as mentioned before, the species of concern is mountain tapir. Only four audio records w/ tapir calls could be obtained from the study site. Consequently, some tapir data collected from Cali Zoo (Zoologico de Cali, Cali, Columbia) were also used.

In this repository, I have summarised the data in CSV files. In addition, I provide links to the drive containing audio clips and spectrogram images. This, however, requires requesting access. The sub-directories cattle_pres and cattle_abs under Cattle directory contain the labeled audio files. Tapir data is under the corresponding directory.


<a name="tex"></a>
## Technologies

[Python 3](https://www.python.org/download/releases/3.0/)

[matplotlib](https://matplotlib.org/) library: Module matplotlib.pyplot for creating, displaying, and saving figures/colorplots.

[pathib](https://pathlib.readthedocs.io/en/pep428/) module: Classes Path and PurePath for working w/ files and directories.

[pandas](https://pandas.pydata.org/) package: For working w/ dataframes.

[Pydub](http://pydub.com/) library: Used in data augmentation. pydub.AudioSegment for extracting samples and metadata from a sound file, splitting an audio file into parts, and merging various files together.

[SciPy](https://scipy.org/) library: Module scipy.signal for creating spectrogram from an audio file.

[torch](https://pypi.org/project/torch/) library : Classes and sub-modules of torch.nn used for writing variational autoencoder; part of [PyTorch](https://pytorch.org/) framework.

[torchvision](https://pytorch.org/vision/stable/index.html) library: Classes of datasets and transforms modules for transforming the dataset to a form accessible for pytorch. 

<!--[OpenSoundscape](http://opensoundscape.org/en/latest/) framework: Corresponding library used for coding CNN models to analyse bio-acoustic data.-->


<!--<a name="exsum"></a>-->
<!--## Executive Summary (or Work summary...)-->
<a name="intro"></a>
## Introduction

The broad goal of the project is to understand the interaction between wildlife and domestic animals like cattle, dogs while also exploring the effect of poaching through gunshots. The machine learning problem in this project pertains to identifying animal species in acoustic data collected as explained earlier. Since deep learning algorithms require a significant amount of training data, there is also a scope of generating synthetic data points for the species which there are insufficient audio clips for.

Ideally, one would expect the model to identify all the species in the given data. However, for simplicity, the classification model in this repository is restricted to presence/absence of one species, which is a binary classification problem. For this purpose, I have focused on cattle. Accordingly, the model needs to be trained with a dataset containing records for the presence and absence of cattle. Here, a 2-dimensional convolutional network is used for this purpose, which performs best on data with spatial features. In an image, nearby pixels are usually related to each other thereby making spectrograms relevant for use w/ a CNN. Additionally, the second problem addressed in this repository pertains to tapir data. Since only a handful of raw audio records w/ tapir sounds could be collected, data augmentation needs to be performed on them, so as to inflate the size of the tapir positive/presence dataset. I have done this using two methods, at the level of audio signals and at the level of spectrograms. This will potentially be useful for future work on tapir absence/presence classification modelling problem.

<a name="spec"></a>
## Spectrogram

A common approach to deep learning with sound data (e.g., in speech recognition) is to generate a spectrogram, which is a diagram showing how frequency varies with time (i.e., a frequency vs time plot) in an acoustic signal. It represents different frequencies in different colours, signifying the amplitude (or loudness) of each frequency in the signal. For the problems in this repository, the goal is not to identify a sequence of letters from a sound (which is so in speech recognition task). Nevertheless spectrograms offer a very useful starting point.

To obtain a spectrogram, an acoustic signal is first divided into a number of segments and each is Fourier transformed. (Fourier transformation is a procedure to decompose a waveform into linear combination of its constituent frequencies: $a_1 f_1 + a_2 f_2 + \cdot\cdot\cdot$, where $a_i$ is the amplitude of the frequency $f_i$.) The Fourier transforms for each segment are combined into a single plot. I've shown a sample spectrogram later.

It turns out that humans perceive a very small range of frequencies, because of which a frequency vs time plot is not very informative. Further, our perception of the difference between two sounds is NOT in terms of the difference between their frequencies, but in terms of the logarithm their ratio. To exemplify, the distinction between two sounds w/ frequencies 10 Hz and 20 Hz is much more than that between sounds of frequencies 100 Hz and 110 Hz. Based on psychoacoustic experiments, there is a consensus that [mel scale](https://en.wikipedia.org/wiki/Mel_scale) offers a good representation of the frequencies that humans typically hear. Similar to frequency, humans also perceive loudness logarithmically, which is quantified using Decibel scale. Our perceptions of frequency and loudness ae accounted for using mel spectrograms, which have mel scale on the y axis and colour signifies decibel levels.

<a name="speccode"></a>
### Code specifics

Mel spectrogram is most commonly generated using librosa (through librosa.feature.melspectrogram() method) or torchaudio (through torchaudio.transforms.MelSpectrogram class) libraries. For the purpose of synthetic audio generation, spectrograms are only used for specifying the temporal location of the tapir call. Accordingly, I've created spectrograms depicting logarithmic frequency against time. This I've achieved using spectrogram() method of scipy.signal module. This method takes a numpy array, created from Audiosegment object for the corresponding audio file, as an argument. An example can be seen in the figure below. Such frequency blobs stacked on top of each other typically correspond to a nasal sound from a mammal.

## Problems with sparse tapir data

As stated before, tapir data will be relevant for future work involving ML modelling for identifying tapir presence/absence in the audio. Since there were four audio files from the study site containing five tapir calls in all, this was certainly insufficient to train the model. Further, the five tapir sounds I had were not all distinct. From [literature](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8065771/), I found that there are more types of tapir calls. While it's possible to generate new records using the existing ones, such data augmentation will not solve the representativeness problem just described. Accordingly, I requested the Fieberg Lab to include some of the tapir audio records obtained from the Cali Zoo.

Cali Zoo has one tapir enclosure (**confirm!**) and the data is collected using three AudioMoth devices. Though these are placed at different locations, their range is long enough to capture tapir sounds from the tapir enclosure. Each of them records and saves audio clips at various times of the day. The clips recorded by two or more AudioMoths at a particular time of the day have the same tapir sound, but with different background on account of the location of each AudioMoth device. While this brings in much more tapir data, for the purpose of representativeness problem discussed in the last paragraph, the number of relevant new clips is just the 'union' of the clips from the 3 AudioMoths (each clip used exactly once). This increased the number of tapir files in the base dataset (which is used to generate new data) to 10.

## Generating synthetic audio clips

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

### Code specifics

As explained before, I have generated synthetic audio clips in two ways. For silence in background, I generated chunks of silence with silent() class method of AudioSegment. For each audio file, I created an AudioSegment instance using from_file('filename') class method, and extracted a chunk of this clip from a to b msec using [a:b] (similar to slicing a NumPy array). For naming new clips generated using a particular clip, I used the stem property of PurePath('existing_clip_name') instance.

## An alternative - controlled spectrogram augmentation

At this point, it is worth noting that in this method of data augmentation, I've worked at the level of audio clips. It is equally possible to enhance the dataset using spectrograms corresponding to the audio clips in the base dataset. Since the spectrogram is a pictorial frequency vs time representation, this involves working with images. One advantage of working with spectrograms in a controlled manner is that both frequency band and time duration of tapir sound are accessible. Such an analysis can be found in the [literature](https://arxiv.org/abs/1904.08779), and exploits both frequency-masking and time-masking. In this repository, I choose to not perform controlled augmentation of spectrogram data since the analysis goes on similar lines as audio augmentation. However, in the next section, I explore the viability of a state of the art deep learning algorithm for enhancing tapir present spectrogram data. This could be useful for those in the larger community who want to do modelling starting from spectrogram images.

## Variational autoencoder to generate synthetic spectrogram images

Variational autoencoder is a neural network based generative (in that it attempts to identify the structure of the data so as to simulate the data generation process), unsupervised (in that it doesn't require class labels for training) algorithm. It was proposed by [Kingma and Welling in 2013](https://doi.org/10.48550/arXiv.1312.6114).

### Introduction

Consider a neural network that applies a set of non-linear transformations to the input data (to reduce its dimension) and maps it to a probability distribution, from which a latent vector is sampled. This network is an encoder. Another neural network, a decoder, then maps the latent vector back to the original input space using non-linear transformations. Essentially, the encoder compresses the data while the decoder decompresses it.
    <p align="center">
    <img src="https://user-images.githubusercontent.com/83636458/201551679-1d44e831-d8a4-4165-96ca-5bc10abef688.png"/>
        <br>
        <em> Representation of a variational autoencoder. Image sourced from [internet](https://avandekleut.github.io/assets/vae/variational-autoencoder.png) </em>
    </p>
As stated before, the input data is encoded as distribution over latent space random variable. The latent space thus obtained is continuous (as opposed to vae's architectural cousin - an autoencoder - where input data is deterministically mapped to the latent space). Further, one of the loss functions to be minimised during training is Kulback-Leibler divergence, which also penalises the distribution for deviating from standard Gaussian. This effectively enforces the requirement that the variance should be close to an identity matrix and mean should be small. This makes the generative process possible.

Let's call the latent random variable $z$. Suppose the encoded distribution is $q_\phi(z|x)$, where $x$ is the training data. Hence, $q_\phi(z|x)$ corresponds to the encoder. Decoder is the likelihood $p_\theta(x|z)$, where $\theta$ represents the parameters of the model. $q_\phi(z|x)$ is the variational approximation to the posterior of the model, and accordingly, $\phi$ represents the variational parameters. Posterior distribution is initially represented by a prior $p(z)$ (which is assumed to be a unit normal distribution N(0,1)), that will be subsequently updated. As mentioned earlier, the latent random variable is sampled from the encoded distribution, i.e. $z ~ \sim ~ q_\phi(z|x)$. This makes the gradient calculation one of the terms in KL divergence (which is required during backpropagation) cumbersome. To overcome this, the authors proposed a reparametrisation trick, which involves expressing the random variable $z$ as a deterministic variable. If it is assumed that the posterior is approximately normal, then so would the variational approximation to the posterior $(q_\phi(z|x))$ be. In this case, the reparameterised latent vector can be represented as $z ~ = ~ \mu ~ + ~ \sigma \cdot \epsilon$, where $\epsilon$ is an auxillary noise variable $\epsilon ~ \sim ~ N(0,1)$.

### Architecture of variational autoencoder

Both encoder and decoder part of the algorithm contain neural network layers. Since I'm working with images, I have employed an architecture containing convolution and pooling layers for changing spatial resolution of the data. For spatially connected data, these layers are superior to a network of fully connected layers for a number of reasons, which I discuss next.

For an image with $\sim ~ 10^2 ~ \times ~ 10^2$ pixels, a fully connected network with $100$ nodes in the hidden layers would have $\sim ~ 10^{10}$ (or $10$ billion) parameters! In a convolution layer, on the other hand, each neuron connects to only its receptive field. In simpler terms, a pixel in any layer is connected to only a local group of pixel in the previous layer. This is called sparse connectivity, and is a reasonable expectation because of locality - nearby pixels are expected to be more informative than a grouping of distant pixels. In addition to this, there is weight sharing within the receptive field. This further reduces the parameters of the model.

In the context of vision architectures, the weights in question are typically arranged in a filter or kernel, which is essentially a $2 \times 2$ matrix (and can be represented as a $2$-d array) for my purpose. For detecting a particular feature (say, an edge), there would be a filter. As explained before, an image is a $2 \times 2$ matrix with each element representing the number of pixels, typically between 0 and 255. Convolution operation is performed between this filter and a patch of image at a time having the size of the filter, covering whole of the image. Each convolution generates a scalar (or a number), leading to a matrix once the whole image is covered. This is called a feature map, and represents features of the image.

The fact that the weights in the filter are learnt during training is what makes convnets so powerful. Essentially, what features are to be extracted are learnt by the algorithm. The feature maps generated by the convolution layer are further passed through a pooling layer. Its purpose is to reduce the size (or dimensionality) of the feature maps. It summarises the features in the input, through averaging, or selecting maximum of each section, et cetera. This has a very important consequence - invariance. While in the original image, the location or orientation of the subject might have been relevant, the summarised feature map is invariant to translation/rotation/deformation.

### Code specifics

Raw audio clips have just 19 tapir sounds. This few data points, however, are abysmally low to train a DNN. Hence, my base dataset for generating synthetic spectrogram images comes from the controlled audio augmentation with background, which was discussed above. Accordingly, I have 95 spectrogram images, which I perform data augmentation on. Using these images, I create an instance of ImageFolder (torchvision library), while transforming them to tensor (which PyTorch works with). ImageFolder instance is further used to create DataLoder instance, which is how data is fed to a PyTorch model.

The architecture I use for an encoder is a set of $6$ layers, each containing a $2$-dimensional convolution layer, a ReLU activation function (to learn non-linearity), a batch normalisation layer (to normalise the data), and a max pooling layer in sequence, followed by fully connected layers for reading distributions for mean and variance of the training data. The image resolution is down-sampled, while increasing the number of channels. Accordingly, each convolution layer has more filters than the previous one. The input data (in the form of DataLoader instance) is given to the encoder, which returns mean and (logarithmic) variance. This is followed by reparameterisation of the latent vector, which is then passed to the decoder to return a reconstructed image. The decoder I've written up-samples the data using $6$ layers, each containing a $2$-d transposed convolution with stride (for up-sampling), a ReLU activation function, and a batch normalisation layer in sequence.

The loss function I've used is the sum of binary cross entropy and KL divergence. I used Adam optimiser, which offers a modified form of gradient descent. I experimented with learning rates of $10^{-3}$, $10^{-4}$, $10^{-5}$, and settled with the last one. Since I have only 95 data points to begin with, I used a small batch size of 32.

There are two ways to use variational autoencoder in the validation phase. One is to use the probability distribution of the input image dataset learnt during training phase to reconstruct the images. The other is to pass Gaussian distribution to the decoder and generate synthetic images. The distribution being passed is for the latent random variable, and is exactly Gaussian, as against the learnt distribution which would be approximately Gaussian. Accordingly, the latter approach generates images somewhat different from the original ones.

## Binary classification

As mentioned before, some of the raw audio files have been labeled as cattle presence/absence (summarised under Data directory). The idea is to use this labelled audio data for training the model. Audio waveform is a time-series. Accordingly, it is natural to consider 1D ConvNets (employing time-only convolutions) for acoustic signals. However, in the last few years, a consensus has been achieved that these models are mostly inferior to (unless made very deep, in that case they could be as good as) 2D vision architectures. Hence, it makes sense to take the latter approach - using 2-dimensional convolutional neural network.

Training a DNN (deep neural networks) typically requires tremendous amount of data and advanced hardware. Here, I'd like to demonstrate transfer learning approach to classification, which is relevant in cases where there is insufficient data to train the model from scratch, or if there isn't a GPU available. Transfer learning approach employs models pre-trained on an exising database. A popular example in literature is [ImageNet](https://image-net.org/index.php), which is a dataset of millions of labeled images. A pre-trained model has already learnt its weights on ImageNet dataset. However, these weights still need to be fine-tuned using data relevant to the problem at hand. <!--Further, I use OpenSoundscape framework... -->


Tune learning rate, batch size. To make this model relevant for tapir data, tune # epochs too since don't have a stopping criterion (no validation set).

1D ConvNets (employing time-only convolutions) have been used directly on acoustic waveform (). However, their performance is subdued (https://arxiv.org/abs/1610.00087) since acoustic signal is a sequence and CNNs don't have the ability to retain information from prior inputs. A CNN-RNN (i.e., a recurrent layer at the end of a convolutional network) has been shown to produce better results. [This article](https://peerj.com/articles/13152.pdf) explains this issues well.
