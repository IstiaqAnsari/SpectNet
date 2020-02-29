import torch
from torch.nn.modules import Module
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary,summary
from math import floor,ceil
import numpy as np, matplotlib.pyplot as plt, pandas as pd, os
def plotf(x):
    plt.plot(x.cpu().detach().numpy())


class MFCC_Gen(nn.Module):
    def __init__(self,kernel_size = 81,filters = 26,fs=1000,winlen=0.025,winstep=0.01,dimension=1):
        super(MFCC_Gen,self).__init__()
        self.gamma = Conv_Gammatone(in_channels=1,out_channels=filters ,kernel_size=kernel_size,fsHz=fs)
#         self.gamma = nn.Conv1d(in_channels=1,out_channels=filters ,kernel_size=81,stride=1)
        self.gammanorm = nn.BatchNorm1d(filters)
        self.mfcc = nn.Conv1d(filters,filters,int(winlen*fs),stride=int(winstep*fs),padding=0,bias=False)
        self.normmfcc = nn.BatchNorm1d(filters)
        self.normmfcc2D = nn.BatchNorm2d(1)
        with torch.no_grad():
            self.mfcc.weight = Parameter(torch.stack([torch.eye(filters) for i in range(int(winlen*fs))],dim=2))
        for x in self.mfcc.named_parameters():
            x[1].requires_grad = False
        for x in self.gamma.named_parameters():
            x[1].requires_grad = False
    def forward(self,x):
        x = self.gamma(x)
        x = self.gammanorm(x)
        x = torch.pow(torch.abs(x),2)
        x = self.mfcc(x)
        x = torch.log(x+0.0000000000000001)
        x = self.normmfcc(x)
        return x


from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _single
class Conv_Gammatone(Module):

    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']

    def __init__(self, in_channels, out_channels, kernel_size,fsHz, stride=1,
                 padding=0, dilation=1, transposed=False, output_padding=(0,),
                 groups=1, bias=False, padding_mode='zeros',fc=None,
                 beta_val=100,amp_val=10**4,n_order=4):
        super(Conv_Gammatone, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filters = out_channels
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride)
        self.padding = _single(padding)
        self.dilation = _single(dilation)
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        self.fsHz = fsHz
        if(fc is not None):
            if isinstance(fc,tuple):
                (minf,maxf)=fc
            else:
                minf = 0
                maxf = fc
        else:
            minf = 0
            maxf = self.fsHz/2
        self.fc = Parameter(torch.from_numpy(self.mel2hz(np.linspace(self.hz2mel(minf),self.hz2mel(maxf),
                            self.filters,dtype=np.float32))).unsqueeze(-1))
        
        self.beta = Parameter(torch.ones((self.filters,1))*beta_val)
        
        self.amp = Parameter(torch.ones((self.filters,1))*amp_val)
        
        self.n_order = (torch.tensor(n_order,dtype=torch.float))
        
#         self.weight = torch.([self.fc, self.beta, self.amp])
        
        self.register_parameter('bias', None)
    def impulse_gammatone(self):
        device = 0
#         print(self.amp.get_device())
#         print(self.beta.get_device())
#         print(self.fc.get_device())
#         print(self.n_order.get_device())
        
        self.t = torch.arange(0,self.kernel_size[0]/self.fsHz,
                            1/self.fsHz,dtype = torch.float32).unsqueeze(-1).transpose(1,0)
    
        self.t = self.t.type(torch.FloatTensor)
        self.n_order = self.n_order.type(torch.FloatTensor)
#         print("device",self.t.get_device())
#         print(self.n_order.get_device())
#         print(self.amp.get_device())
        power = torch.pow(self.t,self.n_order-1)
#         print("power ", power.get_device())
        power = power.to(device = device)
#         print("power ", power.get_device())
        
        at = self.amp.to(device=device)*power
        
#         print("exp")
#         print((-2*torch.tensor(np.pi).to(device)).get_device())
#         print(":/ "torch.mm(self.beta,self.t.to(device)).get_device())
        
        exp = torch.exp(-2*torch.tensor(np.pi).to(device)*torch.mm(self.beta.to(device),self.t.to(device)))
        cos = torch.cos(2*torch.tensor(np.pi).to(device)*torch.mm(self.fc.to(device),self.t.to(device)))
        return at*exp*cos
    def forward(self, input):
        gammatone = self.impulse_gammatone().unsqueeze(1)
#         if self.padding_mode == 'circular':
#             expanded_padding = ((self.padding[0] + 1) // 2, self.padding[0] // 2)
#             return F.conv1d(F.pad(input, expanded_padding, mode='circular'),
#                             gammatone, self.bias, self.stride,
#                             _single(0), self.dilation, self.groups)
        return F.conv1d(input, gammatone, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
    
    
#     def reset_parameters(self):
#         init.kaiming_uniform_(self.weight, a=math.sqrt(5))
#         if self.bias is not None:
#             fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
#             bound = 1 / math.sqrt(fan_in)
#             init.uniform_(self.bias, -bound, bound)
    def hz2mel(self,hz):
        return 2595 * np.log10(1+hz/700.)
    def mel2hz(self,mel):
        return 700*(10**(mel/2595.0)-1)
    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(Conv_Gammatone, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'