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
            self.domain = Domain_classifier(domain_class=domain_class)
            
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
    