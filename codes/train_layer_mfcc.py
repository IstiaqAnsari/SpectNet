from __future__ import print_function, division, absolute_import
# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
# set_session(tf.Session(config=config))
# from clr_callback import CyclicLR
# import dill
from BalancedDannAudioDataGenerator import BalancedAudioDataGenerator
import os,time
from scipy.io import loadmat
import numpy as np
np.random.seed(1)
import math
import pandas as pd
import tables,h5py
from datetime import datetime
import argparse
from keras.utils import plot_model
# from Heartnet import heartnet,getAttentionModel
from collections import Counter
from torchviz import make_dot
# from utils import log_macc, results_log
from dataLoader import reshape_folds
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
import matplotlib.pyplot as plt
# import seaborn as sns
# import Evaluator
import dataLoader
# from custom_layers import Attention
from sklearn.model_selection import train_test_split

class wow():
    def __init__(self):
        self.dann = False
        self.self = True
        self.reduce = None
        self.shuffle = 1
        self.mfcc = False
args = wow()

import h5py
path = '../data/fold0_noFIR.mat'
data = h5py.File(path, 'r')

x_train = data['trainX'][:].astype('float32')
x_train = np.expand_dims(x_train.transpose(),1)

x_val = data['valX'][:].astype('float32')
x_val = np.expand_dims(x_val.transpose(),1)

y_train = data['trainY'][:].astype('int32')
y_train = y_train.transpose()
y_train = y_train[:,0]
y_train[y_train<0] = 0

y_val = data['valY'][:].astype('int32')
y_val = y_val.transpose()
y_val = y_val[:,0]
y_val[y_val<0] = 0

batch_size = 100
datagen_source = BalancedAudioDataGenerator(shift=.1,data_format = 'channels_first')
flow_source = datagen_source.flow(x_train, y_train,
                meta_label=y_train,
                batch_size=batch_size, shuffle=True,
                seed=1)
datagen_val = BalancedAudioDataGenerator(shift=.1,data_format = 'channels_first')
flow_val = datagen_val.flow(x_val, y_val,
                meta_label=y_val,
                batch_size=batch_size, shuffle=True,
                seed=1)

import torch.optim as optim
import torch
from torch.autograd import Variable
import time
import torch
import torch.nn as nn
# from torchvision import datasets, transforms
from torchsummary import summary
import numpy as np
from HeartCepTorch import MFCC_Gen,Network
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
from torchsummary import summary

model = Network(2,0)
mfcc_gen = MFCC_Gen(fs=1000,filters=64)

optimizer = optim.Adam(model.parameters(), lr= .001)
class_criterion = nn.CrossEntropyLoss()

from sklearn.metrics import confusion_matrix
eps = 0.0000001
def log_macc(y_pred,y_pred_domain, y_val,val_parts):
    y_pred = y_pred.cpu().detach().numpy()
    y_pred_domain = y_pred_domain.cpu().detach().numpy()
    y_val = y_val.cpu().detach().numpy()
    true = []
    pred = []
    files = []
    start_idx = 0

    y_pred = np.argmax(y_pred, axis=-1)
    y_val = np.transpose(np.argmax(y_val, axis=-1))

    for j,s in enumerate(val_parts):

        if not s:  ## for e00032 in validation0 there was no cardiac cycle
            continue
        # ~ print "part {} start {} stop {}".format(s,start_idx,start_idx+int(s)-1)

        temp_ = y_val[start_idx:start_idx + int(s)]
        temp = y_pred[start_idx:start_idx + int(s)]

        if (sum(temp == 0) > sum(temp == 1)):
            pred.append(0)
        else:
            pred.append(1)

        if (sum(temp_ == 0) > sum(temp_ == 1)):
            true.append(0)
        else:
            true.append(1)

        if val_files is not None:
            files.append(val_files[start_idx])

        start_idx = start_idx + int(s)
    TN, FP, FN, TP = confusion_matrix(true, pred, labels=[0,1]).ravel()
    # TN = float(TN)
    # TP = float(TP)
    # FP = float(FP)
    # FN = float(FN)
    sensitivity = TP / (TP + FN + eps)
    specificity = TN / (TN + FP + eps)
    precision = TP / (TP + FP + eps)
    F1 = 2 * (precision * sensitivity) / (precision + sensitivity + eps)
    Macc = (sensitivity + specificity) / 2
    
    print("TN:",TN,"FP:",FP,"FN:",FN,"TP:",TP)
    print("Sensitivity:","%.2f"%sensitivity,"Specificity:","%.2f"%specificity,"Precision:","%.2f"%precision)
    print("F1:", "%.2f"%F1,"MACC", "%.2f"%Macc)


model.cuda()
mfcc_gen.cuda()
mfcc_gen.eval()
epochs = 2
print("steps ", flow_source.steps_per_epoch)
for e in range(epochs):
    print("EPOCH   ",e+1)
    model.train()
    epoch_loss = 0
    acc = 0
    N = 0
    for i in range(flow_source.steps_per_epoch+1):
        
        optimizer.zero_grad()
        
        x,y = flow_source.next()
        x,y = torch.from_numpy(x),torch.from_numpy(y)
        x = x.type(torch.FloatTensor).cuda()
        x = mfcc_gen(x)
        x = x.transpose(2,1)
        x = x.unsqueeze(1)
        # print(x.shape)
        x,y = Variable(x),Variable(y)
        
        y = y.long().cuda()        
        cls = model(x)
        # class_loss = class_criterion(cls,torch.argmax(y,axis=1))        
        class_loss = class_criterion(cls,y)
        loss = class_loss
        epoch_loss = epoch_loss + loss
        acc = acc + torch.sum(y==torch.argmax(cls,axis=1))
        N = N+len(y)
        loss.backward()
        optimizer.step()
    print("Training loss", "%.2f"%(epoch_loss.item()/flow_source.steps_per_epoch),end=' ')
    print("Training Acc ", "%.2f"%(acc/N).item(),end=' ')
    # Validate 
    model.eval()
    epoch_loss = 0
    acc = 0
    N = 0
    with torch.no_grad():
        for i in range(flow_val.steps_per_epoch+1):
            x,y = flow_source.next()
            x,y = torch.from_numpy(x),torch.from_numpy(y)
            x = x.type(torch.FloatTensor).cuda()
            x = mfcc_gen(x)
            x = x.transpose(2,1)
            x = x.unsqueeze(1)
            x,y = Variable(x),Variable(y)
            #x = x.reshape(x.shape[0],1,x.shape[1],x.shape[2])
            y = y.long().cuda()
            cls= model(x)
            # val_class_loss = class_criterion(cls,torch.argmax(y,axis=1))
            val_class_loss = class_criterion(cls,y)
            acc = acc + torch.sum(y==torch.argmax(cls,axis=1))
            N = N+len(y)
            epoch_loss = epoch_loss + val_class_loss
            # log_macc(cls,y,val_parts)
        print("Validation loss", "%.2f"%(epoch_loss.item()/flow_val.steps_per_epoch),end=' ')
        print("Validation Acc ", "%.2f"%(acc/N).item(),end=' ')
    flow_val.reset()
    flow_source.reset()