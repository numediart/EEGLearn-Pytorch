# Pytorch - EEGLearn 

This repo describes an implementation of the models described in "Learning Representations from EEG with Deep Recurrent-Convolutional Neural Networks." Bashivan et al. at International conference on learning representations (2016).

EEGLearn is a tool aiming to classify electroencephalogram (EEG) into different classes representing different mental states. The framework specificity is based on the fact that the raw EEG are transformed into images representing spatial (electrodes' position) and frequential (power spectral bands analysis) information in a more understandable way. The pipeline of the implementation is described on the following diagram.

![alt text](diagram.png "Converting EEG recordings to movie snippets")
|:--:| 
| Taken from [Bashivan et al. 2016](https://arxiv.org/pdf/1511.06448.pdf)|


This implementation aiming to present a pytorch implementation of the mentioned works, the functions related to the creation of the images have directly been copied and pasted from [original github](https://github.com/pbashivan/EEGLearn) in the Utils_Bashivan.py script. The rest of the implementation has been totally recoded with Pytorch lib.

## Requirements

In order to run the codes, the following libraries (and their corresponding dependencies) have to been installed:

- Python     3.7
- Pytroch     1.3.1
- Cudatoolkit     10.1.243
- Cudnn     7.6.3

Installation with pip: `pip install -r requirements.txt`

Import of the environment with conda: `conda env create -f Pytorch_EEG.yml`


## Notes 

A [jupyter notebook](EEGLearn_ShortDemo.ipynb) presents a short summary of the codes. Before running it, it is necessary to create the "eeg images" by running the [train](Train.py) script a first time or the create_img() function from [Utils](Utils.py). 

The early stopping being not implemented in Pytorch, a version of the code with a custom version will be uploaded as soon as possible.


## References 

If you are using this code please [cite](Cite.bib) the paper:

Bashivan, et al. "Learning Representations from EEG with Deep Recurrent-Convolutional Neural Networks." International conference on learning representations (2016).

https://arxiv.org/pdf/1511.06448.pdf
