#!/usr/bin/env python
# coding: utf-8

# In[42]:


import math
import torch
from torch import nn
import torch.nn.functional as F
from meta_neural_network_architectures import *


# In[74]:


class BatchNorm(nn.Module):
    def __init__(self, num_features, device, args, eps=1e-5, momentum=0.1):
        super(BatchNorm, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        
    def forward(self, x, num_step):
        if self.training:
            m1 = x.mean((0, 2, 3))
            m2 = (x**2).mean((0, 2, 3))
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * m1
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * (m2 - m1**2)
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, False,
                            self.momentum, self.eps)


# In[2]:


class BottleneckLayer(nn.Module):
    def __init__(self, in_channels, device, args, batch_norm_cls):
        super(BottleneckLayer, self).__init__()
        self.k = 64 # growth rateaa
        self.bn1 = batch_norm_cls(in_channels, device, args)
        self.conv1 = nn.Conv2d(in_channels, 4 * self.k, 1)
        self.bn2 = batch_norm_cls(4 * self.k, device, args)
        self.conv2 = nn.Conv2d(4 * self.k, self.k, 3, padding=1)
    
    def forward(self, x, num_step):
        x = self.conv1(F.relu(self.bn1(x, num_step)))
        return self.conv2(F.relu(self.bn2(x, num_step)))


# In[3]:


class SqueezeExciteConvLayer(nn.Module):
    def __init__(self, in_channels):
        super(SqueezeExciteConvLayer, self).__init__()
        reduced = max(in_channels // 16, 1)
        self.w1 = nn.Linear(in_channels, reduced, bias=False)
        self.w2 = nn.Linear(reduced, in_channels, bias=False)
        
    def forward(self, x):
        z = x.mean((-2, -1))
        z = F.relu(self.w1(z))
        s = torch.sigmoid(self.w2(z)).unsqueeze(2).unsqueeze(3)
        return s * x


class DenseBlockUnit(nn.Module):
    def __init__(self, in_channels, device, args, batch_norm_cls):
        super(DenseBlockUnit, self).__init__()
        self.se = SqueezeExciteConvLayer(in_channels)
        self.bc = BottleneckLayer(in_channels, device, args, batch_norm_cls)
        self.n_out_channels = self.bc.k + in_channels
    
    def forward(self, x, num_step):
        y = self.bc(self.se(x), num_step)
        return torch.cat((x, y), 1)


# In[29]:

class HighEndEmbedding(nn.Module):
    def __init__(self, device, args):
        super(HighEndEmbedding, self).__init__()
        
        self.dbu1 = DenseBlockUnit(3, device, args, BatchNorm)
        self.dbu2 = DenseBlockUnit(self.dbu1.n_out_channels, device, args, BatchNorm)
        
        n_out2 = max(self.dbu2.n_out_channels // 2, 1)
        self.tr_conv = nn.Conv2d(self.dbu2.n_out_channels, n_out2, 1)
        self.tr_av_pool = nn.AvgPool2d(2, stride=2)
        
        self.dbu3 = DenseBlockUnit(n_out2, device, args, BatchNorm)
        self.n_out_channels = self.dbu3.n_out_channels
    
    def forward(self, x):
        x = self.dbu2(self.dbu1(x)) # first dense block
        x = self.tr_av_pool(self.tr_conv(x)) # transition layer
        return self.dbu3(x)


class HighEndClassifier(nn.Module):
    def __init__(self, device, args, in_channels):
        super(HighEndClassifier, self).__init__()
        
        self.dbu4 = DenseBlockUnit(in_channels, MetaBatchNormLayer)
        
        self.lin1 = nn.Linear(self.dbu4.n_out_channels, args['num_classes_per_set'])
    
    def forward(self, x, num_step):
        x = self.dbu4(x, num_step).mean((-2, -1))
        return F.softmax(self.lin1(x), dim=-1)


