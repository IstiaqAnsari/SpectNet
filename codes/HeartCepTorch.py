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
    def __init__(self,kernel_size = 81,filters = 26,fs=1000,winlen=0.025,winstep=0.01,dimension=1,momentum=0.99):
        super(MFCC_Gen,self).__init__()
        self.gamma = Conv_Gammatone(in_channels=1,out_channels=filters ,kernel_size=kernel_size,fsHz=fs)
#         self.gamma = nn.Conv1d(in_channels=1,out_channels=filters ,kernel_size=81,stride=1)
        self.gammanorm = nn.BatchNorm1d(filters,momentum=momentum)
        self.mfcc = nn.Conv1d(filters,filters,int(winlen*fs),stride=int(winstep*fs),padding=0,bias=False)
        self.normmfcc = nn.BatchNorm1d(filters,momentum=momentum)
        self.normmfcc2D = nn.BatchNorm2d(1,momentum=momentum)
        with torch.no_grad():
            self.mfcc.weight = Parameter(torch.stack([torch.eye(filters) for i in range(int(winlen*fs))],dim=2))
        for x in self.mfcc.named_parameters():
            x[1].requires_grad = False
        for x in self.gamma.named_parameters():
            x[1].requires_grad = False
    def forward(self,x):
        x = self.gamma(x)
        # gm = x
        x = self.gammanorm(x)
        # gmnorm = x
        x = torch.pow(torch.abs(x),2)
        x = self.mfcc(x)
        x = torch.log(x+0.0000000000000001)
#         x = x.unsqueeze(1)
        x = self.normmfcc(x)
        return x#,gm,gmnorm

class MFCC_Gen_coeff(nn.Module):
    def __init__(self,kernel_size = 81,filters = 26,fs=1000,winlen=0.025,winstep=0.01,dimension=1,momentum=0.99):
        super(MFCC_Gen_coeff,self).__init__()
        self.gamma = nn.Conv1d(1,64,kernel_size=81)
        wow = Conv_Gammatone_coeff(in_channels=1,out_channels=filters ,kernel_size=kernel_size,fsHz=fs)
        self.gamma.weight = wow.weight
        del wow
#         self.gamma = nn.Conv1d(in_channels=1,out_channels=filters ,kernel_size=81,stride=1)
        self.gammanorm = nn.BatchNorm1d(filters,momentum=momentum)
        self.mfcc = nn.Conv1d(filters,filters,int(winlen*fs),stride=int(winstep*fs),padding=0,bias=False)
        self.normmfcc = nn.BatchNorm1d(filters,momentum=momentum)
        self.normmfcc2D = nn.BatchNorm2d(1,momentum=momentum)
        with torch.no_grad():
            self.mfcc.weight = Parameter(torch.stack([torch.eye(filters) for i in range(int(winlen*fs))],dim=2))
        for x in self.mfcc.named_parameters():
            x[1].requires_grad = False
        for x in self.gamma.named_parameters():
            x[1].requires_grad = False
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

class Conv_Gammatone_coeff(Module):

    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']

    def __init__(self, in_channels, out_channels, kernel_size,fsHz, stride=1,
                 padding=0, dilation=1, transposed=False, output_padding=(0,),
                 groups=1, bias=False, padding_mode='zeros',fc=None,
                 beta_val=100,amp_val=10**4,n_order=4):
        super(Conv_Gammatone_coeff, self).__init__()
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
        self.fc = (torch.from_numpy(self.mel2hz(np.linspace(self.hz2mel(minf),self.hz2mel(maxf),
                            self.filters,dtype=np.float32))).unsqueeze(-1))
        
        self.beta = (torch.ones((self.filters,1))*beta_val)
        
        self.amp = (torch.ones((self.filters,1))*amp_val)
        
        self.n_order = (torch.tensor(n_order,dtype=torch.float))
        
        self.weight = Parameter(self.impulse_gammatone().unsqueeze(1))
        del self.beta,self.amp,self.n_order,self.fc
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
        
#         if self.padding_mode == 'circular':
#             expanded_padding = ((self.padding[0] + 1) // 2, self.padding[0] // 2)
#             return F.conv1d(F.pad(input, expanded_padding, mode='circular'),
#                             gammatone, self.bias, self.stride,
#                             _single(0), self.dilation, self.groups)
        return F.conv1d(input, self.weight, self.bias, self.stride,
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


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
from torchsummary import summary

class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)


class Network(nn.Module):

    def __init__(self,num_class, domain_class,data_format = "time_freq"):
        super(Network, self).__init__()
        self.num_class = num_class
        self.domain_class = domain_class
        self.data_format = data_format
        self.extractor = Extractor(self.data_format)
        self.classifier = Class_classifier(num_class=num_class,in_feature=int(7168))
        if(self.domain_class>0):
            self.domain = Domain_classifier(domain_class=domain_class,in_feature=int(7168))
            
    def forward(self, x, hp_lambda=0):
        x = self.extractor(x)
        clss = self.classifier(x)
        
        if(self.domain_class>0):
            dom = self.domain(x,hp_lambda)
            return clss,dom
        return clss

class Extractor(nn.Module):

    def __init__(self,data_format):

        self.data_format = data_format
        if(self.data_format=="time_freq"):
            self.form = True
        else:
            #"freq_time"
            self.form = False

        super(Extractor, self).__init__()
        self.conv0 = nn.Conv2d(1, 16, kernel_size= ((3,2) if(self.form) else (2,3)) ,stride=1,padding=((3,1) if(self.form) else (1,3)) )   ## change with input shape
        self.bn0 = nn.BatchNorm2d(16)
        
        # Res block 1
        self.conv1 = nn.Conv2d(16, 32, kernel_size=(3,3))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv11 = nn.Conv2d(32, 32, kernel_size=(3,3),stride=(1,1),padding=(2,2))
        self.bn11 = nn.BatchNorm2d(32)
        
        # Res block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,3), stride=(1,1),padding=(1,1))
        self.bn2 = nn.BatchNorm2d(64)
        self.conv21 = nn.Conv2d(64, 64, kernel_size=(3,3), stride=(2,2),padding=(1,1))
        self.bn21 = nn.BatchNorm2d(64)

        # Res block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3,3), stride=(1,1),padding=(1,1))
        self.bn3 = nn.BatchNorm2d(128)
        self.conv31 = nn.Conv2d(128, 128, kernel_size=(3,3), stride=(2,2),padding=(1,1),dilation=((2,1) if(self.form) else (1,2)))
        self.bn31 = nn.BatchNorm2d(128)

        # Res block 4
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3,3), stride=(1,1),padding=(1,1))
        self.bn4 = nn.BatchNorm2d(256)
        self.conv41 = nn.Conv2d(256, 256, kernel_size=(3,3), stride=(2,2),padding=(1,1))
        self.bn41 = nn.BatchNorm2d(256)
        
        # self.conv5 = nn.Conv2d(256, 256, kernel_size=((5,3) if(self.form) else (3,5)), stride=((2,1) if(self.form) else (1,2)),padding=(1,1)) ### change with input shape
        # self.bn5 = nn.BatchNorm2d(256)
        
        self.drop = nn.Dropout2d(0.5)
        
    def forward(self, x):
    
#         print("OK")
        x = F.relu(self.bn0(self.conv0(x)))
        #Res block 1
        x1 = self.drop(F.relu(self.bn1(self.conv1(x))))
        x1 = F.relu(F.max_pool2d(self.drop(self.bn11(self.conv11(x1))), 2))
        x = torch.cat((x,torch.zeros_like(x)), axis=1)
        x = F.max_pool2d(x,2)
        x = x+x1
        
        
        #Res block 2
        x1 = self.drop(F.relu(self.bn2(self.conv2(x))))
        x1 = F.relu(self.drop(self.bn21(self.conv21(x1))))
        x = torch.cat((x,torch.zeros_like(x)), axis=1)
        x = F.max_pool2d(x,2)
        x = x+x1
        #Res block 3
        x1 = self.drop(F.relu(self.bn3(self.conv3(x))))
        x1 = F.relu(self.drop(self.bn31(self.conv31(x1))))
        x = torch.cat((x,torch.zeros_like(x)), axis=1)
        x = F.max_pool2d(x,2)
        x = x+x1
        #Res block 4
        x1 = self.drop(F.relu(self.bn4(self.conv4(x))))
        x1 = F.relu(self.drop(self.bn41(self.conv41(x1))))
        x = torch.cat((x,torch.zeros_like(x)), axis=1)
        x = F.max_pool2d(x,2)
        x = x+x1
        
        #last conv
        # x = self.drop(F.relu(self.bn5(self.conv5(x))))
        x = F.max_pool2d(x,((2,1) if(self.form) else (1,2)))  ### change withinput
        x = x.view(x.size(0),-1)
        return x

class Class_classifier(nn.Module):

    def __init__(self, num_class,in_feature=64*3*15,intermediate_nodes=100):
        super(Class_classifier, self).__init__()
        # self.fc1 = nn.Linear(50 * 4 * 4, 100)
        # self.bn1 = nn.BatchNorm1d(100)
        # self.fc2 = nn.Linear(100, 100)
        # self.bn2 = nn.BatchNorm1d(100)
        # self.fc3 = nn.Linear(100, 10)
        self.fc1 = nn.Linear(in_feature, intermediate_nodes)
        self.fc2 = nn.Linear(intermediate_nodes, num_class)
        
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(dim=1)
        
    def forward(self, x):
        # logits = F.relu(self.bn1(self.fc1(input)))
        # logits = self.fc2(F.dropout(logits))
        # logits = F.relu(self.bn2(logits))
        # logits = self.fc3(logits)
        logits = self.relu(self.fc1(x))
        logits = self.fc2(F.dropout(logits))
        logits = self.soft(logits)

        return logits

class Domain_classifier(nn.Module):

    def __init__(self,domain_class,in_feature=64*3*15,intermediate_nodes=100):
        super(Domain_classifier, self).__init__()
        # self.fc1 = nn.Linear(50 * 4 * 4, 100)
        # self.bn1 = nn.BatchNorm1d(100)
        # self.fc2 = nn.Linear(100, 2)
        
        self.fc1 = nn.Linear(in_feature, intermediate_nodes)
        self.fc2 = nn.Linear(intermediate_nodes, domain_class)
        
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(dim=1)
    def forward(self, x, constant):
        x = GradReverse.grad_reverse(x, constant)
        # logits = F.relu(self.bn1(self.fc1(input)))
        # logits = F.log_softmax(self.fc2(logits), 1)
        logits = self.relu(self.fc1(x))
        logits = self.soft(self.fc2(logits))

        return logits

class Branch(nn.Module):

    def __init__(self,c_in, c_out, kernel_size=5,stride=1,dropout = 0.5):
        super(Branch, self).__init__()
        self.conv1 = nn.Conv1d(c_in, c_out, kernel_size=kernel_size,stride=stride)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(c_out, 2*c_out, kernel_size=kernel_size,stride=stride)
        self.bn2 = nn.BatchNorm1d(c_out*2)
    def forward(self, x):
        x = self.drop(self.relu(self.bn1(self.conv1(x))))
        x = self.drop(self.relu(self.bn2(self.conv2(x))))
        return x

class Res_block(nn.Module):
    def __init__(self,c_in,c_out,kernel_size=5,stride=1,dropout=0.5,padding=2):
        super(Res_block,self).__init__()
        self.conv1 = nn.Conv1d(c_in,c_out,kernel_size=kernel_size,stride=stride,padding=padding)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel_size=kernel_size,stride=1,padding=padding)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.pool = nn.MaxPool1d(stride)
        
        
    def forward(self,x):
        x1 = self.drop(self.relu(self.bn1(self.conv1(x))))
        x1 = self.drop(self.relu(self.bn2(self.conv2(x1))))
        x = self.pool(x)
        x = torch.cat((x,torch.zeros_like(x)), axis=1)
        x = x+x1
        return x

class Smallnet(nn.Module):

    def __init__(self,num_class,domain_class):
        super(Smallnet, self).__init__()
        self.branch1 = Branch(1,8)
        self.res_block1 = Res_block(16,32,stride=2)
        self.res_block2 = Res_block(32,64,stride=2)
        self.pool = nn.MaxPool1d(2)
        self.conv = nn.Conv1d(64,128,kernel_size=5,stride=2)
        self.drop = nn.Dropout(0.5)
        
        self.classifier = Class_classifier(num_class=num_class,in_feature=9856,intermediate_nodes=50)
        self.domain = Domain_classifier(domain_class=domain_class,in_feature=9856,intermediate_nodes=80)
    def forward(self, x, hp_lambda=0):
        x = self.branch1(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.drop(self.pool(self.conv(self.pool(x))))
        x = x.view(-1,9856)
        clss = self.classifier(x)
        dom = self.domain(x,hp_lambda)
        return clss,dom
    