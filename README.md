# SpectNet : End-to-End Audio Signal Classification using Learnable Spectrogram Features

Pattern recognition from audio signals is an active research topic encompass-
ing audio tagging, acoustic scene classification, music classification, and other
areas. Spectrogram and mel-frequency cepstral coefficients (MFCC) are among
the most commonly used features for audio signal analysis and classification. Re-
cently, deep convolutional neural networks (CNN) have been successfully used
for audio classification problems using spectrogram-based 2D features. In this
paper, we present SpectNet, an integrated front-end layer that extracts spec-
trogram features within a CNN architecture that can be used for audio pattern
recognition tasks. The front-end layer utilizes learnable gammatone filters that
are initialized using mel-scale filters. The proposed layer outputs a 2D spectro-
gram image which can be fed into a 2D CNN for classification. The parameters
of the entire network, including the front-end filterbank, can be updated via
back-propagation. This training scheme allows for fine-tuning the spectrogram-
image features according to the target audio dataset. The proposed method is
evaluated in two different audio signal classification tasks: heart sound anomaly
detection and acoustic scene classification. The proposed method shows a sig-
nificant 1.02% improvement in MACC for the heart sound classification task
and 2.11% improvement in accuracy for the acoustic scene classification task
compared to the classical spectrogram image features

## The Proposed  front-end layer - learnable filterbank
The front-end layers provides control over the number of filterbanks, range of frequnecy of the signal domain and the order of the filters. 
The layers can be found in [this](codes/HeartCepTorch.py). The MFCC_gen module generates MFCC features using a Convolutional Gammatone Filterbank. By default the the center frequencies are chosen according to the MEL Scale within the range defined. It can be set using any distribution or constant values also. 

![The front-end layer](images/frontend.png)

## Experiments | Datasets | Results

We use the **Physionet HeartSound Dataset** for experiments on Heart Sound signal. Data distribution are given below:
|Subset | Total Subject | Normal recordings |Abnormal recordings|Used device|
|:-------:|:---------------:|:------------------:|:-----------------:|:---------------:|
|a |121| 117| 292| Welch Allyn Meditron|
|b| 106| 385| 104| 3M Littmann E4000|
|c| 31| 7| 24| AUDIOSCOPE|
|d| 38| 27| 28| Infral Corp. Prototype|
|e (Norm.)| 174| 1867| 0| MLT201/Piezo|
|e (Abn.)| 3352| 0| 151| 3M Littmann|
|f| 112| 80| 34| JABES|
||

We experimented with multiple setups. The **Static** keyword means the front-end layers was kept static AKA non-learnable or default MFCC feature. Our proposed method let's the front-end layer fine-tune the filterbank parameters. 

Performance Comparison on Physionet Heart Sound Classification Task

|Method| Accuracy| F1| Macc| Sensitivity| Specificity| Precision|
|:------:|:--------:|:-----:|:-----:|:-----------:|:------------:|:----------:|
||Baseline|Methods|||
|Gammatone 1D-CNN| 75.80| 83.17| 82.30| 91.30| 73.29| 76.36|
|Gammatone 2D-CNN| 76.69| 84.67| 84.03| 92.03| 76.03| 78.39|
||SpectNet| (Static |SpectNet)||
SpectNet-4 + ResNet| 77.42| 81.91| 80.30| 93.48| 67.12| 72.88|
SpectNet-8 + ResNet| 81.64| 87.27| 87.66| 86.96| 88.36| 87.59|
SpectNet-16 + ResNet| 81.79| 87.37| 87.68| 87.68| 87.67| 87.05|
||Proposed| System |(Learnable | SpectNet)
|**SpectNet-16 + ResNet**| 80.36| 88.32| **88.70**| 87.68| 89.73| 88.97|
|||||


We also experimented with the DCASE dataset. The DCASE 2016 acoustic scene classification challenge dataset consists of audio samples from 15 (fifteen) different indoor and outdoor locations or envi-
ronments. These are Beach, Bus, Cafe/Restaurant, Car, City Center, Forest
Path, Grocery Store, Home, Library, Metro Station, Office, Park, Residential
Area, Train, and Tram. There are 1170 and 390 audio segments in the training
and validation dataset, respectively. Each class has 234 samples for training
and 78 samples for validation. 

The results on this datasets are as follows:

|Methods| Accuracy| Sensitivity| Specificity| Precision| F1| MACC|
|:------:|:--------:|:--------:|:----------:|:--------:|:-:|:----:|
|||Static| SpectNet|
|SpectNet-16 + CNN| 66.92| 65.44| 97.53| 65.88| 65.66| 81.47|
|SpectNet-32 + CNN| 73.59| 73.59| 98.11| 74.37| 73.98| 85.85|
|SpectNet-46 + CNN| 74.44| 74.50| 98.18| 74.95| 74.10| 86.34|
|SpectNet-64 + CNN| 73.93| 73.56| 98.11| 74.05| 73.81| 85.84|
|SpectNet-128 + CNN| 75.55| 75.56| 98.25| 76.10| 75.83| 86.91|
|SpectNet-149 + CNN| 74.81 |74.22| 98.16| 74.94| 74.58| 86.19|
||**Proposed**| **System**| **(Learnable**| **SpectNet)**|
|SpectNet-46 + CNN| **76.55**| 76.47| 98.32| 77.15| 76.07| 87.39|
||

## Hardware
Gpu titan Xp, Gpu driver 430.50, Cuda driver 10.1 ,Cuda runtime library 10.0 ,Cudnn library 7.4.1


Cudnn conda 7.6.1 (came with tensorflow-gpu), tensorflow 1.13.1

