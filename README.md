# Pytorch - EEGLearn 

This repo describes an implementation of the models described in "Learning Representations from EEG with Deep Recurrent-Convolutional Neural Networks." Bashivan et al. at International conference on learning representations (2016).

EEGLearn is a tool aiming to classify electroencephalogram (EEG) into different classes representing different mental states. The framework specificity is based on the fact that the raw EEG are transformed into images representing spatial (electrodes' position) and frequential (power spectral bands analysis) information in a more understandable way. The pipeline of the implementation is described on the following diagram.

![alt text](diagram.png "Converting EEG recordings to movie snippets")
|:--:| 
| *Space* |

All the codes have been inspired from the [original github](https://github.com/pbashivan/EEGLearn).


