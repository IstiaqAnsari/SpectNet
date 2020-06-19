import torch
from torch.nn.modules import Module
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary,summary
from math import floor,ceil
import numpy as np, matplotlib.pyplot as plt, pandas as pd, os
from torch.nn.parameter import Parameter
from HeartCepTorch import Conv_Gammatone_coeff, Network

class WHOLE_NEW_MODEL(nn.Module):
    def __init__(self,kernel_size = 81,filters = 64,fs=1000,winlen=0.025,winstep=0.01,dimension=1,momentum=0.99):
        super(WHOLE_NEW_MODEL,self).__init__()
        self.gamma = nn.Conv1d(1,64,kernel_size=81)
        wow = Conv_Gammatone_coeff(in_channels=1,out_channels=filters ,kernel_size=kernel_size,fsHz=fs)
        self.gamma.weight = wow.weight
        del wow
        self.gammanorm = nn.BatchNorm1d(filters,momentum=momentum)
        self.mfcc = nn.Conv1d(filters,filters,int(winlen*fs),stride=int(winstep*fs),padding=0,bias=False)
        self.normmfcc = nn.BatchNorm1d(filters,momentum=momentum)
        self.normmfcc2D = nn.BatchNorm2d(1,momentum=momentum)
        with torch.no_grad():
            self.mfcc.weight = Parameter(torch.stack([torch.eye(filters) for i in range(int(winlen*fs))],dim=2))
        for x in self.mfcc.named_parameters():
            x[1].requires_grad = False
            
        self.classifier = Network(2,0)
    def forward(self,x):
        x = self.gamma(x)
        # gm = x
        x = self.gammanorm(x)
        # gmnorm = x
        x = torch.pow(torch.abs(x),2)
        x = self.mfcc(x)
        x = torch.log(x+0.0000000000000001)
        # x = x.unsqueeze(1)
        x = self.normmfcc(x)
        
        x = x.unsqueeze(1)
        x = x.transpose(3,2)
        
        x = self.classifier(x)
        
        return x