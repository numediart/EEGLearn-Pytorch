'''
Created by Victor Delvigne
ISIA Lab, Faculty of Engineering University of Mons, Mons (Belgium)
victor.delvigne@umons.ac.be

Source: Bashivan, et al."Learning Representations from EEG with Deep Recurrent-Convolutional Neural Networks." International conference on learning representations (2016).

Copyright (C) 2019 - UMons

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
'''

import numpy as np
import scipy.io as sio
import torch
import os
from os import path

import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader,random_split

from Utils import *
from Models import *

torch.manual_seed(1234)
np.random.seed(1234)

import warnings
warnings.simplefilter("ignore")


if not path.exists("Sample Data/images_time.mat"):
    print("Time Windom Images didn't exist need to be created.")
    create_img()

Images = sio.loadmat("Sample Data/images_time.mat")["img"] #corresponding to the images mean for all the seven windows
Mean_Images = np.mean(Images, axis= 0)
#Mean_Images = sio.loadmat("Sample Data/images.mat")["img"] #corresponding to the images mean for all the seven windows
Label = (sio.loadmat("Sample Data/FeatureMat_timeWin")["features"][:,-1]-1).astype(int) #corresponding to the signal label (i.e. load levels).
Patient_id = sio.loadmat("Sample Data/trials_subNums.mat")['subjectNum'][0] #corresponding to the patient id

# Introduction: training a simple CNN with the mean of the images.
train_part = 0.8
test_part = 0.2

batch_size = 32
choosen_patient = 9
n_epoch = 30
n_rep = 20

for patient in np.unique(Patient_id):

    Result = []
    for r in range(n_rep):
        EEG = EEGImagesDataset(label=Label[Patient_id == patient], image=Mean_Images[Patient_id == patient])
        lengths = [int(len(EEG) * train_part), int(len(EEG) * test_part)]
        if sum(lengths) != len(EEG):
            lengths[0] = lengths[0] + 1
        Train, Test = random_split(EEG, lengths)
        Trainloader = DataLoader(Train, batch_size=batch_size)
        Testloader = DataLoader(Test, batch_size=batch_size)
        res = TrainTest_Model(BasicCNN, Trainloader, Testloader, n_epoch=n_epoch, learning_rate=0.001, print_epoch=-1,
                              opti='Adam')
        Result.append(res)
    sio.savemat("Res_Basic_Patient"+str(patient)+".mat", {"res":Result})
    Result = np.mean(Result, axis=0)
    print ('-'*100)
    print('\nBegin Training for Patient '+str(patient))
    print('End Training with \t loss: %.3f\tAccuracy : %.3f\t\tval-loss: %.3f\tval-Accuracy : %.3f' %
          (Result[0], Result[1], Result[2], Result[3]))
    print('\n'+'-'*100)

print("\n\n\n\n Maxpool CNN \n\n\n\n")

for patient in np.unique(Patient_id):

    Result = []
    for r in range(n_rep):
        EEG = EEGImagesDataset(label=Label[Patient_id == patient], image=Images[Patient_id == patient])
        lengths = [int(len(EEG) * train_part + 1), int(len(EEG) * test_part)]
        if sum(lengths) < len(EEG):
            lengths[0] = lengths[0] + 1
        if sum(lengths) > len(EEG):
            lengths[0] = lengths[0] - 1
        Train, Test = random_split(EEG, lengths)
        Trainloader = DataLoader(Train, batch_size=batch_size)
        Testloader = DataLoader(Test, batch_size=batch_size)
        res = TrainTest_Model(MaxCNN, Trainloader, Testloader, n_epoch=n_epoch, learning_rate=0.001, print_epoch=-1,
                              opti='Adam')
        Result.append(res)
    sio.savemat("Res_MaxCNN_Patient"+str(patient)+".mat", {"res":Result})
    Result = np.mean(Result, axis=0)
    print ('-'*100)
    print('\nBegin Training for Patient '+str(patient))
    print('End Training with \t loss: %.3f\tAccuracy : %.3f\t\tval-loss: %.3f\tval-Accuracy : %.3f' %
          (Result[0], Result[1], Result[2], Result[3]))
    print('\n'+'-'*100)

print("\n\n\n\n Temp CNN \n\n\n\n")


for patient in np.unique(Patient_id):

    Result = []
    for r in range(n_rep):
        EEG = EEGImagesDataset(label=Label[Patient_id == patient], image=Images[Patient_id == patient])
        lengths = [int(len(EEG) * train_part + 1), int(len(EEG) * test_part)]
        if sum(lengths) < len(EEG):
            lengths[0] = lengths[0] + 1
        if sum(lengths) > len(EEG):
            lengths[0] = lengths[0] - 1
        Train, Test = random_split(EEG, lengths)
        Trainloader = DataLoader(Train, batch_size=batch_size)
        Testloader = DataLoader(Test, batch_size=batch_size)
        res = TrainTest_Model(TempCNN, Trainloader, Testloader, n_epoch=n_epoch, learning_rate=0.001, print_epoch=-1,
                              opti='Adam')
        Result.append(res)
    sio.savemat("Res_TempCNN_Patient"+str(patient)+".mat", {"res":Result})
    Result = np.mean(Result, axis=0)
    print ('-'*100)
    print('\nBegin Training for Patient '+str(patient))
    print('End Training with \t loss: %.3f\tAccuracy : %.3f\t\tval-loss: %.3f\tval-Accuracy : %.3f' %
          (Result[0], Result[1], Result[2], Result[3]))
    print('\n'+'-'*100)


print("\n\n\n\n LSTM CNN \n\n\n\n")


for patient in np.unique(Patient_id):

    Result = []
    for r in range(n_rep):
        EEG = EEGImagesDataset(label=Label[Patient_id == patient], image=Images[Patient_id == patient])
        lengths = [int(len(EEG) * train_part + 1), int(len(EEG) * test_part)]
        if sum(lengths) < len(EEG):
            lengths[0] = lengths[0] + 1
        if sum(lengths) > len(EEG):
            lengths[0] = lengths[0] - 1
        Train, Test = random_split(EEG, lengths)
        Trainloader = DataLoader(Train, batch_size=batch_size)
        Testloader = DataLoader(Test, batch_size=batch_size)
        res = TrainTest_Model(LSTM, Trainloader, Testloader, n_epoch=n_epoch, learning_rate=0.001, print_epoch=1,
                              opti='Adam')
        Result.append(res)
    sio.savemat("Res_LSTM_Patient"+str(patient)+".mat", {"res":Result})
    Result = np.mean(Result, axis=0)
    print ('-'*100)
    print('\nBegin Training for Patient '+str(patient))
    print('End Training with \t loss: %.3f\tAccuracy : %.3f\t\tval-loss: %.3f\tval-Accuracy : %.3f' %
          (Result[0], Result[1], Result[2], Result[3]))
    print('\n'+'-'*100)


print("\n\n\n\n Mix CNN \n\n\n\n")


for patient in np.unique(Patient_id):

    Result = []
    for r in range(n_rep):
        EEG = EEGImagesDataset(label=Label[Patient_id == patient], image=Images[Patient_id == patient])
        lengths = [int(len(EEG) * train_part + 1), int(len(EEG) * test_part)]
        if sum(lengths) < len(EEG):
            lengths[0] = lengths[0] + 1
        if sum(lengths) > len(EEG):
            lengths[0] = lengths[0] - 1
        Train, Test = random_split(EEG, lengths)
        Trainloader = DataLoader(Train, batch_size=batch_size)
        Testloader = DataLoader(Test, batch_size=batch_size)
        res = TrainTest_Model(Mix, Trainloader, Testloader, n_epoch=n_epoch, learning_rate=0.001, print_epoch=-1,
                              opti='Adam')
        Result.append(res)
    sio.savemat("Res_Mix_Patient"+str(patient)+".mat", {"res":Result})
    Result = np.mean(Result, axis=0)
    print ('-'*100)
    print('\nBegin Training for Patient '+str(patient))
    print('End Training with \t loss: %.3f\tAccuracy : %.3f\t\tval-loss: %.3f\tval-Accuracy : %.3f' %
          (Result[0], Result[1], Result[2], Result[3]))
    print('\n'+'-'*100)