import matplotlib.pyplot as plt,h5py,numpy as np
from collections import Counter

class Data():
    def __init__(self,path,f,n,severe = False):
        self.data = h5py.File(path+f, 'r')
        self.file = f
        self.dom = n
        self.trainX = np.array(self.data['trainX'][:]).astype('float32')
        self.trainY = self.data['trainY'][:][0].astype('int8')
        self.train_parts = self.data['train_parts'][0].astype('int32')
        self.valX = None
        self.valY = None
        self.normfiles, self.abnormfiles = self.parts()
        self.domainY = [n]*self.trainY.shape[0]
        
        self.normal = Counter(self.trainY)[0]
        self.abnormal = Counter(self.trainY)[1]
        self.total = self.normal+self.abnormal
        
        if(f[:3]=='com'):
            self.valX = self.data['valX'][:]
            self.valY = self.data['valY'][:][0]
            self.valY[self.valY>0] = 1
            if(severe):
                self.sevX = None
                self.sevY = self.trainY[self.trainY<2]
                self.sev_parts = []
                left = int(0)
                for x in d.train_parts:
                    x = int(x)
                    if(all([i==2 for i in d.trainY[left:x+left]])):
                        if(self.sevX is None):
                            self.sevX = self.trainX[:,left:x+left]
                        else: 
                            self.sevX = np.concatenate((self.sevX,self.trainX[:,left:x+left]),axis=1)
                        
                    else:self.sev_parts.append(x)
                    left = x + left
            else:
                self.trainY[self.trainY>0] = 1
        elif(f[:3]=='pas'):
            self.trainY[self.trainY<0] = 0
        else:
            self.trainY[self.trainY<0] = 0
        if('fold_e' in f):
            print(self.trainY.shape)
            nn = ab = 0
            idx = []
            left = 0
            parts = []
            for x in self.train_parts:
                if(all([(l == 0) for l in self.trainY[left:left+x]])):
                    if(nn<self.abnormal):
                        idx = idx + [True]*x
                        nn = nn + x
                        parts.append(x)
                    else:idx = idx + [False]*x
                else:
                    if(ab<self.abnormal):
                        idx = idx + [True]*x
                        ab = ab + x
                        parts.append(x)
                    else:idx = idx + [False]*x
                left = left + x
            self.trainY = self.trainY[idx]
            self.trainX = np.transpose(np.transpose(self.trainX)[idx])
            self.train_parts = parts
            self.total = nn+ab
            self.domainY = self.domainY[:self.total]
            print(self.trainY.shape)
    def parts(self):
        y = 0
        nn = 0
        ab = 0
        for x in self.train_parts:
            if(sum(self.trainY[y:y+int(x)])>0):
                ab = ab + 1
            else: nn = nn +1 
            y = int(x)+y
        #print(nn+ab,nn,ab)
        return nn,ab
    def pie(self):
        colors = ['gold','lightskyblue']
        explode = (0.07, 0)
        size = [self.normal,self.abnormal]
        labels = ['N', 'Ab']
        plt.pie(size,labels=labels,colors = colors,startangle=140,explode=explode,
               shadow=True,autopct=self.value)
        plt.title(self.dom)
    def value(self,val):
        return int(self.total*val/100)