import os,tables,numpy as np,matplotlib.pyplot as plt,pandas as pd,h5py,json
from scipy.io import loadmat
from collections import Counter
from  dataholder import Data

class DataMerge():
    def __init__(self,split = 0):
        self.split = split
        self.x_train = self.y_train = self.y_domain = self.train_parts = None
        self.x_val = self.y_val = self.y_valdom = self.val_parts = None
        self.val_wav_name = None
        #self.train_normal, self.train_abnormal, self.val_normal, self.val_abnormal = 0
        #self.train_total , self.val_total = 0
    def merge(self,data,train_test):
        if(train_test):
            if(self.x_val is None):self.x_val = data.trainX;
            else:self.x_val = np.concatenate((self.x_val,data.trainX),axis = 1)
            if(self.y_val is None):self.y_val = data.trainY
            else:self.y_val = np.concatenate((self.y_val,data.trainY),axis = 0)
            if(self.y_valdom is None):self.y_valdom = data.domainY
            else:self.y_valdom = self.y_valdom+data.domainY
            if(self.val_parts is None):
                if(data.train_parts is not None):self.val_parts = data.train_parts
            else:
                if(data.train_parts is not None):
                    self.val_parts = np.concatenate((self.val_parts,data.train_parts),axis = 0)
                else:
                    print("Data train parts unavailable")
            if(self.val_wav_name is None):
                self.val_wav_name = data.wav_name
            else:
                self.val_wav_name = self.val_wav_name+data.wav_name
            if(self.split>0):        
                if(self.x_train is None):self.x_train = data.valX
                else:self.x_train = np.concatenate((self.x_train,data.valX),axis = 1)
                if(self.y_train is None):self.y_train =  data.valY
                else:self.y_train = np.concatenate((self.y_train,data.valY),axis = 0)
                if(self.y_domain is None):self.y_domain  =  data.valdomY
                else:self.y_domain = self.y_domain+data.valdomY
                if(self.train_parts is None):self.train_parts = data.val_parts;
                else:
                    if(data.val_parts is not None):
                        self.train_parts = np.concatenate((self.train_parts,data.val_parts),axis = 0) 
                    else:
                        print("Data train parts unavailable")

        else:
            
            if(self.x_train is None):self.x_train = data.trainX;
            else:self.x_train = np.concatenate((self.x_train,data.trainX),axis = 1)
            if(self.y_train is None):self.y_train = data.trainY;
            else:self.y_train = np.concatenate((self.y_train,data.trainY),axis = 0)
            if(self.y_domain is None):self.y_domain = data.domainY;
            else:self.y_domain = self.y_domain+data.domainY
            if(self.train_parts is None):
                self.train_parts = data.train_parts;
            else:
                if(data.train_parts is not None):
                    self.train_parts = np.concatenate((self.train_parts,data.train_parts),axis = 0) 
                else:
                    print("Data train parts nai Train mergee ")

    def showDistribution(self):
        self.train_normal = Counter(self.y_train)[0]
        self.train_abnormal = Counter(self.y_train)[1]
        self.train_total = self.train_normal+self.train_abnormal

        self.val_normal = Counter(self.y_val)[0]
        self.val_abnormal = Counter(self.y_val)[1]
        self.val_total = self.val_normal+self.val_abnormal
        print("Train normal - ", self.train_normal,"-",self.train_abnormal," Abnormal")
        print("               ", int(100*self.train_normal/self.train_total) , " - ", int(100*self.train_abnormal/self.train_total), "%")
        print("Test normal - ", self.val_normal,"-",self.val_abnormal," Abnormal")
        print("              ",int(100*self.val_normal/self.val_total) , " - ", int(100*self.val_abnormal/self.val_total), "%")


def getData(fold_dir, train_folds, test_folds, split = 0, shuffle = None):
    try:
        with open('../data/domain_filename.json', 'r') as fp:
            foldname = json.load(fp)
    except:
        raise FileNotFoundError("The json file in Data folder of the repository, that maps domain character to filename is not here")

    allData = DataMerge(split)
    for c in test_folds:
        allData.merge(Data(fold_dir,foldname[c],c,severe = False,split=split,shuffle=shuffle),True)
    for c in train_folds:
        allData.merge(Data(fold_dir,foldname[c],c,shuffle=shuffle),False)
    allData.showDistribution()
    return allData.x_train, allData.y_train, allData.y_domain, allData.train_parts,allData.x_val,allData.y_val,allData.y_valdom,allData.val_parts,allData.val_wav_name


def reshape_folds(x, y):
    x_train = []
    for x1 in x:
        x1 = np.transpose(x1[:, :])
        x1 = np.reshape(x1, [x1.shape[0], x1.shape[1], 1])
        x_train.append(x1)
        print("reshaped x ", x1.shape)
    y_train = []
    for y1 in y:
        y1 = np.reshape(y1, [y1.shape[0], 1])
        y_train.append(y1)
        print("reshaped Y ", y1.shape)

    return x_train,y_train

