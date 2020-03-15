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

import torch

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


class BasicCNN(nn.Module):
    '''
    Build the  Mean Basic model performing a classification with CNN 

    param input_image: list of EEG image [batch_size, n_window, n_channel, h, w]
    param kernel: kernel size used for the convolutional layers
    param stride: stride apply during the convolutions
    param padding: padding used during the convolutions
    param max_kernel: kernel used for the maxpooling steps
    param n_classes: number of classes
    return x: output of the last layers after the log softmax
    '''
    def __init__(self, input_image=torch.zeros(1, 3, 32, 32), kernel=(3,3), stride=1, padding=1,max_kernel=(2,2), n_classes=4):
        super(BasicCNN, self).__init__()

        n_channel = input_image.shape[1]

        self.conv1 = nn.Conv2d(n_channel,32,kernel,stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(32,32,kernel,stride=stride, padding=padding)
        self.conv3 = nn.Conv2d(32,32,kernel,stride=stride, padding=padding)
        self.conv4 = nn.Conv2d(32,32,kernel,stride=stride, padding=padding)
        self.pool1 = nn.MaxPool2d(max_kernel)
        self.conv5 = nn.Conv2d(32,64,kernel,stride=stride,padding=padding)
        self.conv6 = nn.Conv2d(64,64,kernel,stride=stride,padding=padding)
        self.conv7 = nn.Conv2d(64,128,kernel,stride=stride,padding=padding)

        self.pool = nn.MaxPool2d((1,1))
        self.drop = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(2048,512)
        self.fc2 = nn.Linear(512,n_classes)
        self.max = nn.LogSoftmax()
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool1(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool1(x)
        x = F.relu(self.conv7(x))
        x = self.pool1(x)
        x = x.reshape(x.shape[0],x.shape[1], -1)
        x = self.pool(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.max(x)
        return x


class MaxCNN(nn.Module):
    '''
    Build the Max-pooling model performing a maxpool over the 7 parallel convnets

    param input_image: list of EEG image [batch_size, n_window, n_channel, h, w]
    param kernel: kernel size used for the convolutional layers
    param stride: stride apply during the convolutions
    param padding: padding used during the convolutions
    param max_kernel: kernel used for the maxpooling steps
    param n_classes: number of classes
    return x: output of the last layers after the log softmax
    '''
    def __init__(self, input_image=torch.zeros(1, 7, 3, 32, 32), kernel=(3,3), stride=1, padding=1,max_kernel=(2,2), n_classes=4):
        super(MaxCNN, self).__init__()

        n_window = input_image.shape[1]
        n_channel = input_image.shape[2]

        self.conv1 = nn.Conv2d(n_channel,32,kernel,stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(32,32,kernel,stride=stride, padding=padding)
        self.conv3 = nn.Conv2d(32,32,kernel,stride=stride, padding=padding)
        self.conv4 = nn.Conv2d(32,32,kernel,stride=stride, padding=padding)
        self.pool1 = nn.MaxPool2d(max_kernel)
        self.conv5 = nn.Conv2d(32,64,kernel,stride=stride,padding=padding)
        self.conv6 = nn.Conv2d(64,64,kernel,stride=stride,padding=padding)
        self.conv7 = nn.Conv2d(64,128,kernel,stride=stride,padding=padding)

        self.pool = nn.MaxPool2d((n_window,1))
        self.drop = nn.Dropout(p=0.5)

        self.fc = nn.Linear(n_window*int(4*4*128/n_window),512)
        self.fc2 = nn.Linear(512,n_classes)
        self.max = nn.LogSoftmax()

    def forward(self, x):
        if x.get_device() == 0:
            tmp = torch.zeros(x.shape[0],x.shape[1],128,4,4).cuda()
        else:
            tmp = torch.zeros(x.shape[0],x.shape[1],128,4,4).cpu()
        for i in range(7):
            tmp[:,i] = self.pool1( F.relu(self.conv7(self.pool1(F.relu(self.conv6(F.relu(self.conv5(self.pool1( F.relu(self.conv4(F.relu(self.conv3( F.relu(self.conv2(F.relu(self.conv1(x[:,i])))))))))))))))))
        x = tmp.reshape(x.shape[0], x.shape[1],4*128*4,1)
        x = self.pool(x)
        x = x.view(x.shape[0],-1)
        x = self.fc2(self.fc(x))
        x = self.max(x)
        return x


class TempCNN(nn.Module):
    '''
    Build the Conv1D model performing a convolution1D over the 7 parallel convnets

    param input_image: list of EEG image [batch_size, n_window, n_channel, h, w]
    param kernel: kernel size used for the convolutional layers
    param stride: stride apply during the convolutions
    param padding: padding used during the convolutions
    param max_kernel: kernel used for the maxpooling steps
    param n_classes: number of classes
    return x: output of the last layers after the log softmax
    '''
    def __init__(self, input_image=torch.zeros(1, 7, 3, 32, 32), kernel=(3,3), stride=1, padding=1,max_kernel=(2,2), n_classes=4):
        super(TempCNN, self).__init__()

        n_window = input_image.shape[1]
        n_channel = input_image.shape[2]

        self.conv1 = nn.Conv2d(n_channel,32,kernel,stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(32,32,kernel,stride=stride, padding=padding)
        self.conv3 = nn.Conv2d(32,32,kernel,stride=stride, padding=padding)
        self.conv4 = nn.Conv2d(32,32,kernel,stride=stride, padding=padding)
        self.pool1 = nn.MaxPool2d(max_kernel)
        self.conv5 = nn.Conv2d(32,64,kernel,stride=stride,padding=padding)
        self.conv6 = nn.Conv2d(64,64,kernel,stride=stride,padding=padding)
        self.conv7 = nn.Conv2d(64,128,kernel,stride=stride,padding=padding)

        #Temporal CNN Layer
        self.conv8 = nn.Conv1d(n_window,64,(4*4*128,3),stride=stride,padding=padding)

        self.pool = nn.MaxPool2d((n_window,1))
        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear(64*3,n_classes)
        self.max = nn.LogSoftmax()

    def forward(self, x):
        if x.get_device() == 0:
            tmp = torch.zeros(x.shape[0],x.shape[1],128,4,4).cuda()
        else:
            tmp = torch.zeros(x.shape[0],x.shape[1],128,4,4).cpu()
        for i in range(7):
            tmp[:,i] = self.pool1( F.relu(self.conv7(self.pool1(F.relu(self.conv6(F.relu(self.conv5(self.pool1( F.relu(self.conv4(F.relu(self.conv3( F.relu(self.conv2(F.relu(self.conv1(x[:,i])))))))))))))))))
        x = tmp.reshape(x.shape[0], x.shape[1],4*128*4,1)
        x = F.relu(self.conv8(x))
        x = x.view(x.shape[0],-1)
        x = self.fc(x)
        x = self.max(x)
        return x


class LSTM(nn.Module):
    '''
    Build the LSTM model applying a RNN over the 7 parallel convnets outputs

    param input_image: list of EEG image [batch_size, n_window, n_channel, h, w]
    param kernel: kernel size used for the convolutional layers
    param stride: stride apply during the convolutions
    param padding: padding used during the convolutions
    param max_kernel: kernel used for the maxpooling steps
    param n_classes: number of classes
    param n_units: number of units
    return x: output of the last layers after the log softmax
    '''
    def __init__(self, input_image=torch.zeros(1, 7, 3, 32, 32), kernel=(3,3), stride=1, padding=1,max_kernel=(2,2), n_classes=4, n_units=128):
        super(LSTM, self).__init__()

        n_window = input_image.shape[1]
        n_channel = input_image.shape[2]

        self.conv1 = nn.Conv2d(n_channel,32,kernel,stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(32,32,kernel,stride=stride, padding=padding)
        self.conv3 = nn.Conv2d(32,32,kernel,stride=stride, padding=padding)
        self.conv4 = nn.Conv2d(32,32,kernel,stride=stride, padding=padding)
        self.pool1 = nn.MaxPool2d(max_kernel)
        self.conv5 = nn.Conv2d(32,64,kernel,stride=stride,padding=padding)
        self.conv6 = nn.Conv2d(64,64,kernel,stride=stride,padding=padding)
        self.conv7 = nn.Conv2d(64,128,kernel,stride=stride,padding=padding)

        # LSTM Layer
        self.rnn = nn.RNN(4*4*128, n_units, n_window)
        self.rnn_out = torch.zeros(2, 7, 128)

        self.pool = nn.MaxPool2d((n_window,1))
        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear(896, n_classes)
        self.max = nn.LogSoftmax()

    def forward(self, x):
        if x.get_device() == 0:
            tmp = torch.zeros(x.shape[0], x.shape[1], 128, 4, 4).cuda()
        else:
            tmp = torch.zeros(x.shape[0], x.shape[1], 128, 4, 4).cpu()
        for i in range(7):
            img = x[:, i]
            img = F.relu(self.conv1(img))
            img = F.relu(self.conv2(img))
            img = F.relu(self.conv3(img))
            img = F.relu(self.conv4(img))
            img = self.pool1(img)
            img = F.relu(self.conv5(img))
            img = F.relu(self.conv6(img))
            img = self.pool1(img)
            img = F.relu(self.conv7(img))
            tmp[:, i] = self.pool1(img)
            del img
        x = tmp.reshape(x.shape[0], x.shape[1], 4 * 128 * 4)
        del tmp
        self.rnn_out, _ = self.rnn(x)
        x = self.rnn_out.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.max(x)
        return x


class Mix(nn.Module):
    '''
        Build the LSTM model applying a RNN and a CNN over the 7 parallel convnets outputs

        param input_image: list of EEG image [batch_size, n_window, n_channel, h, w]
        param kernel: kernel size used for the convolutional layers
        param stride: stride apply during the convolutions
        param padding: padding used during the convolutions
        param max_kernel: kernel used for the maxpooling steps
        param n_classes: number of classes
        param n_units: number of units
        return x: output of the last layers after the log softmax
        '''
    def __init__(self, input_image=torch.zeros(1, 7, 3, 32, 32), kernel=(3,3), stride=1, padding=1,max_kernel=(2,2), n_classes=4, n_units=128):
        super(Mix, self).__init__()

        n_window = input_image.shape[1]
        n_channel = input_image.shape[2]

        self.conv1 = nn.Conv2d(n_channel,32,kernel,stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(32,32,kernel,stride=stride, padding=padding)
        self.conv3 = nn.Conv2d(32,32,kernel,stride=stride, padding=padding)
        self.conv4 = nn.Conv2d(32,32,kernel,stride=stride, padding=padding)
        self.pool1 = nn.MaxPool2d(max_kernel)
        self.conv5 = nn.Conv2d(32,64,kernel,stride=stride,padding=padding)
        self.conv6 = nn.Conv2d(64,64,kernel,stride=stride,padding=padding)
        self.conv7 = nn.Conv2d(64,128,kernel,stride=stride,padding=padding)

        # LSTM Layer
        self.rnn = nn.RNN(4*4*128, n_units, n_window)
        self.rnn_out = torch.zeros(2, 7, 128)

        # Temporal CNN Layer
        self.conv8 = nn.Conv1d(n_window, 64, (4 * 4 * 128, 3), stride=stride, padding=padding)

        self.pool = nn.MaxPool2d((n_window, 1))
        self.drop = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(1088,512)
        self.fc2 = nn.Linear(512, n_classes)
        self.max = nn.LogSoftmax()


    def forward(self, x):
        if x.get_device() == 0:
            tmp = torch.zeros(x.shape[0], x.shape[1], 128, 4, 4).cuda()
        else:
            tmp = torch.zeros(x.shape[0], x.shape[1], 128, 4, 4).cpu()
        for i in range(7):
            img = x[:, i]
            img = F.relu(self.conv1(img))
            img = F.relu(self.conv2(img))
            img = F.relu(self.conv3(img))
            img = F.relu(self.conv4(img))
            img = self.pool1(img)
            img = F.relu(self.conv5(img))
            img = F.relu(self.conv6(img))
            img = self.pool1(img)
            img = F.relu(self.conv7(img))
            tmp[:, i] = self.pool1(img)
            del img

        temp_conv = F.relu(self.conv8(tmp.reshape(x.shape[0], x.shape[1], 4 * 128 * 4, 1)))
        temp_conv = temp_conv.reshape(temp_conv.shape[0], -1)

        self.lstm_out, _ = self.rnn(tmp.reshape(x.shape[0], x.shape[1], 4 * 128 * 4))
        del tmp
        lstm = self.lstm_out.view(x.shape[0], -1)

        x = torch.cat((temp_conv, lstm), 1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.max(x)
        return x

