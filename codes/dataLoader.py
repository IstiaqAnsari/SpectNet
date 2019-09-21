import os,tables,numpy as np,matplotlib.pyplot as plt,pandas as pd,h5py,json
from scipy.io import loadmat
from collections import Counter
from  dataholder import Data

class DataMerge():
    def __init__(self):
        self.x_train = self.y_train = self.y_domain = self.train_parts = None
    def merge(self,data):
        if(self.x_train is None):
            self.x_train = data.trainX;
            self.y_train = data.trainY;
            self.y_domain = data.domainY;
            self.train_parts = data.train_parts;
        else:
            self.x_train = np.concatenate((self.x_train,data.trainX),axis = 1)
            self.y_train = np.concatenate((self.y_train,data.trainY),axis = 0)
            self.y_domain = self.y_domain+data.domainY
            self.train_parts = np.concatenate((self.train_parts,data.train_parts),axis = 0)            

def getData(fold_dir, folds):
    try:
        with open('../data/domain_filename.json', 'r') as fp:
            foldname = json.load(fp)
    except:
        raise FileNotFoundError("The json file that maps domain character to filename is not here")
        
    allData = DataMerge()
    for c in folds:
        allData.merge(Data(fold_dir,foldname[c],c))
    return allData.x_train, allData.y_train, allData.y_domain, allData.train_parts
def reshape_folds(x_train, y_train,x_val, y_val):
    x1 = np.reshape(x_train, [x_train.shape[0], x_train.shape[1], 1])
    y_train = np.reshape(y_train, [y_train.shape[0], 1])

    print(x1.shape)
    print(y_train.shape)
    if(x_val is not None):
        v1 = np.transpose(x_val[:, :])

        v1 = np.reshape(v1, [v1.shape[0], v1.shape[1], 1])

        y_val = np.reshape(y_val, [y_val.shape[0], 1])

        print(v1.shape)
        print(y_val.shape)
        return x1, y_train, v1 , y_val
    return x1,y_train