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

from torch.utils.data.dataset import Dataset
from Utils_Bashivan import *

import torch

import scipy.io as sio
import torch.optim as optim
import torch.nn as nn
import numpy as np

def kfold(length, n_fold):
    tot_id = np.arange(length)
    np.random.shuffle(tot_id)
    len_fold = int(length/n_fold)
    train_id = []
    test_id = []
    for i in range(n_fold):
        test_id.append(tot_id[i*len_fold:(i+1)*len_fold])
        train_id.append(np.hstack([tot_id[0:i*len_fold],tot_id[(i+1)*len_fold:-1]]))
    return train_id, test_id


class EEGImagesDataset(Dataset):
    """EEGLearn Images Dataset from EEG."""
    
    def __init__(self, label, image):
        self.label = label
        self.Images = image
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.Images[idx]
        label = self.label[idx]
        sample = (image, label)
        
        return sample



def Test_Model(net, Testloader, criterion, is_cuda=True):
    running_loss = 0.0 
    evaluation = []
    for i, data in enumerate(Testloader, 0):
        input_img, labels = data
        input_img = input_img.to(torch.float32)
        if is_cuda:
            input_img = input_img.cuda()
        outputs = net(input_img)
        _, predicted = torch.max(outputs.cpu().data, 1)
        evaluation.append((predicted==labels).tolist())
        loss = criterion(outputs, labels.cuda())
        running_loss += loss.item()
    running_loss = running_loss/(i+1)
    evaluation = [item for sublist in evaluation for item in sublist]
    running_acc = sum(evaluation)/len(evaluation)
    return running_loss, running_acc


def TrainTest_Model(model, trainloader, testloader, n_epoch=30, opti='SGD', learning_rate=0.0001, is_cuda=True, print_epoch =5, verbose=False):
    if is_cuda:
        net = model().cuda()
    else :
        net = model()
        
    criterion = nn.CrossEntropyLoss()
    
    if opti=='SGD':
        optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    elif opti =='Adam':
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    else: 
        print("Optimizer: "+optim+" not implemented.")
    
    for epoch in range(n_epoch):
        running_loss = 0.0
        evaluation = []
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs.to(torch.float32).cuda())
            _, predicted = torch.max(outputs.cpu().data, 1)
            evaluation.append((predicted==labels).tolist())
            loss = criterion(outputs, labels.cuda())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        running_loss = running_loss/(i+1)
        evaluation = [item for sublist in evaluation for item in sublist]
        running_acc = sum(evaluation)/len(evaluation)
        validation_loss, validation_acc = Test_Model(net, testloader, criterion,True)
        
        if epoch%print_epoch==(print_epoch-1):
            print('[%d, %3d]\tloss: %.3f\tAccuracy : %.3f\t\tval-loss: %.3f\tval-Accuracy : %.3f' %
             (epoch+1, n_epoch, running_loss, running_acc, validation_loss, validation_acc))
    if verbose:
        print('Finished Training \n loss: %.3f\tAccuracy : %.3f\t\tval-loss: %.3f\tval-Accuracy : %.3f' %
                 (running_loss, running_acc, validation_loss,validation_acc))
    
    return (running_loss, running_acc, validation_loss,validation_acc)



def create_img():
    feats = sio.loadmat('Sample Data/FeatureMat_timeWin.mat')['features']
    locs = sio.loadmat('Sample Data/Neuroscan_locs_orig.mat')
    locs_3d = locs['A']
    locs_2d = []
    # Convert to 2D
    for e in locs_3d:
        locs_2d.append(azim_proj(e))

    images_timewin = np.array([gen_images(np.array(locs_2d),
                                          feats[:, i * 192:(i + 1) * 192], 32, normalize=True) for i in
                               range(int(feats.shape[1] / 192))
                               ])

    sio.savemat("Sample Data/images_time.mat",{"img":images_timewin})
    print("Images Created and Save in Sample Dat/images_time")