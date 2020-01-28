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
from Heartnet import heartnet,getAttentionModel
from collections import Counter
from torchviz import make_dot

from utils import log_macc, results_log
from dataLoader import reshape_folds
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
import matplotlib.pyplot as plt
# import seaborn as sns
import Evaluator
import dataLoader
from custom_layers import Attention
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser(description='Specify fold to process')
parser.add_argument("test_domains",
                    help="which fold to use from balanced folds generated in /media/taufiq/Data/"
                         "heart_sound/feature/potes_1DCNN/balancedCV/folds/")
parser.add_argument("--self",type=bool, help = "If true model train and tests on same data with split")
parser.add_argument("--dann",type=bool, help = "train with or without dann")
parser.add_argument("--train_domains",help = "trainer domain ")
arguments = parser.parse_args()


scheme = 'new'
class wow():
    def __init__(self):
        self.dann = False
        self.self = False
        self.reduce = None
        self.shuffle = 1
        self.mfcc = False
args = wow()
if(arguments.self):
    args.self = arguments.self



test_domains = arguments.test_domains
if(arguments.train_domains is not None):
    train_domains = arguments.train_domains
else:
    train_domains = 'bcdefh'
source_domain = train_domains
target_domain = test_domains

test_split = 0
fold_dir = '../../feature/potes_1DCNN/balancedCV/folds/all_folds_wav_name/'

if(args.self == True):
    print("Self training activated")
    x_train, y_train, y_domain, train_parts, x_val, y_val, val_domain, val_parts,val_wav_files = dataLoader.getData(fold_dir,'',test_domains,0.9,shuffle=args.shuffle)
    print(x_train.shape, x_val.shape)
else:
    x_train, y_train, y_domain, train_parts,x_val, y_val, val_domain, val_parts, val_wav_files = dataLoader.getData(fold_dir,train_domains,test_domains,test_split,shuffle = args.shuffle)

if(args.reduce):
    print("Reduction ", args.reduce)
    x_train,_,y_train,_,y_domain,_ = train_test_split(x_train.transpose(),y_train,y_domain,stratify=y_train,test_size = args.reduce)
    x_train = x_train.transpose()

    #x_val,_,y_val,_,val_domain,_ = train_test_split(x_val.transpose(),y_val,val_domain,stratify=y_val,test_size = args.reduce)
    #x_val = x_val.transpose()

val_files = val_domain
#Create meta labels and domain labels

if(test_split>0):
    source_domain = "".join(set(source_domain).union(set(target_domain)))
    #domains = domains + test_domains

if(args.self):
    print("self training")
    source_domain = test_domains

domains = set(source_domain + target_domain)
#num_class_domain = len(set(train_domains + test_domains))
num_class_domain = len(domains)
num_class = 2

domainClass_source = [(cls,dfc) for cls in range(2) for dfc in source_domain]
domainClass_target = [(cls,dfc) for cls in range(2) for dfc in target_domain]


## Convert to MFCC
import python_speech_features as psf
from matplotlib import cm
if(args.mfcc):
    print("Converting to MFCC")
    train_mfcc = np.array([(psf.base.mfcc(x,samplerate=1000,winlen=0.05,winstep=0.01)) for x in x_train.transpose()])
    val_mfcc = np.array([(psf.base.mfcc(x,samplerate=1000,winlen=0.05,winstep=0.01)) for x in x_val.transpose()])
    
    train_mfcc = (train_mfcc-np.mean(train_mfcc))/np.std(train_mfcc)
    val_mfcc = (val_mfcc-np.mean(val_mfcc))/np.std(val_mfcc)
    #train_mfcc = train_mfcc/np.max(np.abs(train_mfcc))
    #val_mfcc = val_mfcc/np.max(np.abs(val_mfcc))
    
    del x_train, x_val
    x_train = train_mfcc.copy()
    x_val = val_mfcc.copy()
    print(x_train.shape, x_val.shape)

## Get meta labesls and reshape data
meta_labels_source = [domainClass_source.index((cl,df)) for (cl,df) in zip(y_train,y_domain)]
meta_labels_target = None
if(args.dann):
    meta_labels_target = [domainClass_target.index((cl,df)) for (cl,df) in zip((y_val),(val_domain))]
    

domains = "".join(set(source_domain).union(set(target_domain)))

y_domain_source = np.array([list(domains).index(lab) for lab in y_domain])

y_domain_target = np.array([list(domains).index(lab) for lab in val_domain])

################### Reshaping ############

if(args.mfcc):
    [], [y_train,y_domain,y_val] = reshape_folds([],[y_train,y_domain_source,y_val])
else:
    [x_train,x_val], [y_train,y_domain,y_val] = reshape_folds([x_train,x_val],[y_train,y_domain_source,y_val])
y_train = to_categorical(y_train, num_classes=num_class)

print("Y domain ", Counter([x[0] for x in y_domain]))
print("Val domain ", Counter(val_domain))
print("Source Meta labels ", Counter(meta_labels_source))
print("Target Meta labels ", Counter(meta_labels_target))
y_domain_source = to_categorical(y_domain_source,num_classes=num_class_domain)

y_val = to_categorical(y_val, num_classes=num_class)
y_domain_target = to_categorical(y_domain_target,num_classes=num_class_domain)


val_domain = y_domain_target
print("Train files ", y_train.shape, "  Domain ", y_domain.shape)
print("Test files ", y_val.shape, "  Domain ", val_domain.shape)

### Batch Size limmiter 
batch_size = 1000
if(batch_size > max(y_train.shape)):
    print("Batch size if given greater than train files size. limiting batch size")
    batch_size = max(y_train.shape)


######## change 2500 axis for pytorch 
if(not args.mfcc):
    x_train = x_train.transpose((0,2,1))
    x_val = x_val.transpose((0,2,1))


batch_size = 200

flow_target = None
if(args.dann):
    datagen_target = BalancedAudioDataGenerator(shift=.1)
    flow_target = datagen_target.flow(x_val, [y_val,y_domain_target],
                meta_label=meta_labels_target,
                batch_size=batch_size, shuffle=True,
                seed=1)
datagen_source = BalancedAudioDataGenerator(shift=.1)
flow_source = datagen_source.flow(x_train, [y_train,y_domain_source],
                meta_label=meta_labels_source,
                batch_size=batch_size, shuffle=True,
                seed=1)




import torch.optim as optim
import torch
from torch.autograd import Variable
import time
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchsummary import summary
from CSVLogger import CSVLogger
from mfcc_models import Network,Smallnet


model = Smallnet(num_class,num_class_domain,scheme)
optimizer = optim.Adam(model.parameters(), lr= .001)
class_criterion = nn.CrossEntropyLoss()
domain_criterion = nn.CrossEntropyLoss()

keys = ['epoch',
 'class_loss',
 'domain_loss',
 'loss',
 'model_path',
 'val_class_acc',
 'val_class_loss',
 'val_domain_loss',
 'val_loss',
 'val_macc',
 'val_precision',
 'val_sensitivity',
 'val_specificity',
 'val_F1']

from sklearn.metrics import confusion_matrix
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
    print("Sensitivity:","%.2f"%sensitivity,"Specificity:","%.2f"%specificity,"Precision:","%.2f"%precision,end=' ')
    print("F1:", "%.2f"%F1,"MACC", "%.2f"%Macc)
    dic = {'val_macc'}
    return Macc,F1,sensitivity,specificity,precision

from datetime import datetime
log_dir = '/media/mhealthra2/Data/heart_sound/Adversarial Heart Sound Results/torch/logs/'
network = 'Smallnet'
fold_name = test_domains+ (("_"+train_domains) if(not args.self) else " ") + str(datetime.now())
print(fold_name)


log_dir = os.path.join(log_dir,network)
if(scheme == 'new'):
    log_dir = os.path.join(log_dir,'emd')
if(args.mfcc):log_dir = os.path.join(log_dir,'mfcc')
if(args.self):
    print(args.self)
    print("selfing")
    log_dir = os.path.join(log_dir,'self')
if(args.dann):log_dir = os.path.join(log_dir,'dann')
log_dir = os.path.join(log_dir,fold_name)
print(log_dir)
if(not os.path.isdir(log_dir)):
    os.mkdir(log_dir)
logger = CSVLogger(os.path.join(log_dir,'training.csv'))





model.cuda()
epochs = 400

print("steps ", flow_source.steps_per_epoch)


logger.train_begin()
for e in range(epochs):
    logger.epoch_begin(e)
    print("EPOCH   ",e+1)
    model.train()
    epoch_loss = 0
    epoch_class_loss = 0
    epoch_domain_loss = 0
    
    for i in range(flow_source.steps_per_epoch+1):
        
        optimizer.zero_grad()
        
        x,[y,yd] = flow_source.next()
        x,y,yd = torch.from_numpy(x),torch.from_numpy(y),torch.from_numpy(yd)
        x,y,yd = Variable(x),Variable(y),Variable(yd)
        x = x.type(torch.FloatTensor).cuda()
        if(args.mfcc):
            x = x.reshape(x.shape[0],1,x.shape[1],x.shape[2])
        y = y.long().cuda()
        yd = yd.long().cuda()
        # print("soruce",x.shape)
        cls, dom = model(x,hp_lambda=0.8)
        class_loss = class_criterion(cls,torch.argmax(y,axis=1))
        domain_loss_source = domain_criterion(dom,torch.argmax(yd,axis=1))
        
        if(args.dann):
            x,[y,yd] = flow_target.next()
            x,y,yd = torch.from_numpy(x),torch.from_numpy(y),torch.from_numpy(yd)
            x,y,yd = Variable(x),Variable(y),Variable(yd)
            x = x.type(torch.FloatTensor).cuda()
            if(args.mfcc):
                x = x.reshape(x.shape[0],1,x.shape[1],x.shape[2])
            y = y.long().cuda()
            yd = yd.long().cuda()
            #print("taret",x.shape)
            cls, dom = model(x,hp_lambda=0.8)
            domain_loss_target = domain_criterion(dom,torch.argmax(yd,axis=1))
            loss = class_loss + domain_loss_source+domain_loss_target
            epoch_domain_loss = epoch_domain_loss + domain_loss_source+domain_loss_target
        else:
            loss = class_loss
        if(scheme=='new'):
            class_loss.backward(retain_graph=True)
            newloss=torch.mean(domain_loss_source - class_loss)*.5
            newloss.backward()
        else:
            epoch_loss = epoch_loss + loss
            epoch_class_loss = epoch_class_loss + class_loss
        
            loss.backward()
        optimizer.step()
    epoch_domain_loss = epoch_domain_loss/flow_source.steps_per_epoch
    epoch_loss = epoch_loss/flow_source.steps_per_epoch
    epoch_class_loss = epoch_class_loss/flow_source.steps_per_epoch
    
    
    
    print("Training loss", "%.2f"%(epoch_loss.item()),end=' ')
    # Validate 
    model.eval()
    with torch.no_grad():
        cls_pred = None
        dom_pred = None
        epoch_val_loss = 0
        epoch_val_class_loss = 0
        epoch_val_domain_loss = 0
        s = 0
        for part in val_parts:
            x,y,yd = torch.from_numpy(x_val[s:s+part]),torch.from_numpy(y_val[s:s+part]),torch.from_numpy(val_domain[s:s+part])
            s = s + part
            x,y,yd = Variable(x),Variable(y),Variable(yd)
            x = x.type(torch.FloatTensor).cuda()
            if(args.mfcc):
                x = x.reshape(x.shape[0],1,x.shape[1],x.shape[2])
            y = y.long().cuda()
            yd = yd.long().cuda()
            
            #print("hwat",x.shape)
            cls, dom = model(x)
            if(cls_pred is None):
                cls_pred = cls
                dom_pred = dom
            else:
                cls_pred = torch.cat((cls_pred,cls),axis=0)
                dom_pred = torch.cat((dom_pred,dom),axis=0)
            val_class_loss = class_criterion(cls,torch.argmax(y,axis=1))
            val_domain_loss = domain_criterion(dom,torch.argmax(yd,axis=1))
            epoch_val_domain_loss = epoch_val_domain_loss + val_domain_loss
            epoch_val_class_loss = epoch_val_class_loss + val_class_loss
        epoch_val_class_loss = epoch_val_class_loss/len(val_parts)
        epoch_val_domain_loss = epoch_val_domain_loss/len(val_parts)
        print("val_Class_loss  ","%.2f"%epoch_val_class_loss.item())
        print("val_dom_loss    ", "%.2f"%epoch_val_domain_loss.item())
        Macc,F1,sensitivity,specificity,precision = log_macc(cls_pred,dom_pred,torch.from_numpy(y_val).long().cuda(),val_parts)
        
    logger.log('class_loss',epoch_class_loss)
    logger.log('domain_loss',epoch_domain_loss)
    logger.log('loss',epoch_loss)
    logger.log('val_class_loss',val_class_loss)
    logger.log('val_domain_loss',val_domain_loss)
    logger.log('val_loss',val_domain_loss+val_class_loss)
    logger.log('val_macc',Macc)
    logger.log('val_precision',precision)
    logger.log('val_sensitivity',sensitivity)
    logger.log('val_specificity',specificity)
    logger.log('val_F1',F1)
    
    
    flow_source.reset()
    logger.epoch_end(e)  
logger.on_train_end()