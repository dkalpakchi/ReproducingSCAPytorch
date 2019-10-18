#!/usr/bin/env python
# coding: utf-8

# In[101]:


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, n_classes=5):
        super(Critic, self).__init__()
        # self.__args = args
        self.__n_classes = n_classes
        # self.__n_theta = n_theta
        self.__kernel_size = 2
        
        for i in range(5):
            setattr(self, f'_conv{i}', nn.Conv1d(
                in_channels=i * 8 + 1,
                out_channels=8,
                kernel_size=self.__kernel_size,
                dilation=2 ** i,
                padding=math.floor(2 ** (i - 1) * (self.__kernel_size - 1))
            )) # padding?, activation function?
            
        out = 5 * 8 + 1
        self._fc1 = nn.Linear(out * self.__n_classes, out * self.__n_classes)
        self._fc2 = nn.Linear(out * self.__n_classes, 1)
        
        
    def conv_forward(self, x):
        x = x.unsqueeze(1) # add channels
        x1 = F.pad(x, (0, 1)) # really?
        for i in range(5):
            layer = getattr(self, f'_conv{i}')
            y = F.relu(layer(x1 if i == 0 else x))
            x = torch.cat((x, y), 1)
        return x.flatten(1)
        
    def forward(self, predictions):
        x = predictions # TODO: add theta and maybe g?
        x = self.conv_forward(x)
        x = F.relu(self._fc1(x))
        return self._fc2(x)



