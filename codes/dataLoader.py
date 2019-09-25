import os,tables,numpy as np,matplotlib.pyplot as plt,pandas as pd,h5py,json
from scipy.io import loadmat
from collections import Counter
from  dataholder import Data

class DataMerge():
    def __init__(self):
        self.x_train = self.y_train = self.y_domain = self.train_parts = None
        self.x_val = self.y_val = self.y_valdom = self.val_parts = None
    def merge(self,data,train_test):
        if(train_test):
            if(self.x_val is None):
                self.x_val = data.trainX;
                self.y_val = data.trainY;
                self.y_valdom = data.domainY;
                self.val_parts = data.train_parts;
            else:
                self.x_val = np.concatenate((self.x_val,data.trainX),axis = 1)
                self.y_val = np.concatenate((self.y_val,data.trainY),axis = 0)
                self.y_valdom = self.y_valdom+data.domainY
                if(data.train_parts is not None):
                    self.val_parts = np.concatenate((self.val_parts,data.train_parts),axis = 0)
            if(self.x_train is None):
                self.x_train = data.valX
                self.y_train =  data.valY
                self.y_domain =  data.valdomY
                self.train_parts = data.val_parts;
            else:
                self.x_train = np.concatenate((self.x_train,data.valX),axis = 1)
                self.y_train = np.concatenate((self.y_train,data.valY),axis = 0)
                self.y_domain = self.y_domain+data.valdomY
                if(data.train_parts is not None):
                    self.train_parts = np.concatenate((self.train_parts,data.val_parts),axis = 0) 
                    
        else:
            if(self.x_train is None):
                self.x_train = data.trainX;
                self.y_train = data.trainY;
                self.y_domain = data.domainY;
                self.train_parts = data.train_parts;
            else:
                self.x_train = np.concatenate((self.x_train,data.trainX),axis = 1)
                self.y_train = np.concatenate((self.y_train,data.trainY),axis = 0)
                self.y_domain = self.y_domain+data.domainY
                if(data.train_parts is not None):
                    self.train_parts = np.concatenate((self.train_parts,data.train_parts),axis = 0) 

def getData(fold_dir, train_folds, test_folds, split = 0):
    try:
        with open('../data/domain_filename.json', 'r') as fp:
            foldname = json.load(fp)
    except:
        raise FileNotFoundError("The json file that maps domain character to filename is not here")

    allData = DataMerge()
    for c in test_folds:
        allData.merge(Data(fold_dir,foldname[c],c,severe = False,split=split),True)
    for c in train_folds:
        allData.merge(Data(fold_dir,foldname[c],c),False)
        
    return allData.x_train, allData.y_train, allData.y_domain, allData.train_parts,allData.x_val,allData.y_val,allData.y_valdom,allData.val_parts 


def reshape_folds(x, y):
    x_train = []
    for x1 in x:
        x1 = np.transpose(x1[:, :])
        x1 = np.reshape(x1, [x1.shape[0], x1.shape[1], 1])
        x_train.append(x1)
        print(x1.shape)
    y_train = []
    for y1 in y:
        y1 = np.reshape(y1, [y1.shape[0], 1])
        y_train.append(y1)
        print(y1.shape)

    return x_train,y_train