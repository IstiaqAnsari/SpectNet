import matplotlib.pyplot as plt,h5py,numpy as np,random
from collections import Counter
class Data():
    def __init__(self,path,f,n,severe = True,split=0,normalize=False,shuffle=None):
        
        if(split>=1 or split<0):
            print("make sure split follow  1<split<=0 ")
            raise ValueError
        self.split = split
        self.data = h5py.File(path+f, 'r')
        self.file = f
        self.dom = n
        self.segments = ['s1','systole','s2','diastole']
        print(path)
        self.seg = ('4_segments' in path)
        if(self.seg):
            self.trainX = {k:np.array(self.data[k][:]).astype('float32') for k in self.segments}
        else:
            self.trainX = np.array(self.data['trainX'][:]).astype('float32')
        self.trainY = self.data['trainY'][:][0].astype('int8')
        self.train_parts = self.data['train_parts'][0].astype('int32')
        self.wav_name = [''.join([chr(c[0]) for c in self.data[stp]]) for stp in self.data['wav_name'][0]]
        self.valX = None
        self.valY = None
        self.val_wav_name = None
        self.valdomY = None
        self.val_parts = None
        self.domainY = [n]*self.trainY.shape[0]
        ##calculate the normal and abnormal beats count
        self.normal = Counter(self.trainY)[0]
        self.abnormal = Counter(self.trainY)[1]
        self.total = self.normal+self.abnormal
        self.normfiles, self.abnormfiles = self.parts() 
        
        if(f[:3]=='com'):self.processCompare(severe) ### select severe files only
        else:
            self.trainY[self.trainY<0] = 0   
        if('fold_e' in f):self.processE()
        if(shuffle is not None):self.shuffle_data(shuffle)
        if(split>0):self.split_data(split)
        if(normalize):self.normalize_data()
        if(False):self.cutoff()
    def shuffle_data(self,seed=0):
        ## The shuffle parameter is None in init(). but if provided and int value it will be
        ## used as the random seed
        if(self.trainX is not None):
            xx = self.trainX.copy()
            yy = self.trainY.copy()
            pr = [x for x in range(len(self.train_parts))]
            random.Random(seed).shuffle(pr)
            s = 0
            for i,x in enumerate(pr):
                if(self.seg):
                    for k in self.segments:
                        xx[k][:,s:s+self.train_parts[x]] = self.trainX[k][:,sum(self.train_parts[:x]):sum(self.train_parts[:x])+self.train_parts[x]]
                else:
                    xx[:,s:s+self.train_parts[x]] = self.trainX[:,sum(self.train_parts[:x]):sum(self.train_parts[:x])+self.train_parts[x]]
                yy[s:s+self.train_parts[x]] = self.trainY[sum(self.train_parts[:x]):sum(self.train_parts[:x])+self.train_parts[x]]
                s = s+self.train_parts[x]
            self.train_parts = self.train_parts[pr]
            self.wav_name = [self.wav_name[pr[x]] for x in range(len(self.wav_name))]
        if(self.valX is not None):
            xx = self.valX.copy()
            yy = self.valY.copy()
            pr = [x for x in range(len(self.val_parts))]
            random.Random(seed).shuffle(pr)
            s = 0
            for i,x in enumerate(pr):
                if(self.seg):
                    xx[k][:,s:s+self.train_parts[x]] = self.valX[k][:,sum(self.train_parts[:x]):sum(self.train_parts[:x])+self.train_parts[x]]
                else:
                    xx[:,s:s+self.train_parts[x]] = self.valX[:,sum(self.train_parts[:x]):sum(self.train_parts[:x])+self.train_parts[x]]
                yy[s:s+self.train_parts[x]] = self.valY[sum(self.train_parts[:x]):sum(self.train_parts[:x])+self.train_parts[x]]
                s = s+self.train_parts[x]
            self.train_parts = self.val_parts[pr]
            self.val_wav_name = [self.val_wav_name[pr[x]] for x in range(len(self.val_wav_name))]
    def normalize_data(self):
        self.trainX = np.array([x/(max(abs(x)+10e-6)) for x in self.trainX.transpose()]).transpose()
    def cutoff(self):
        self.trainX = np.array([self.cut(x)  for x in self.trainX.transpose()]).transpose()
    def cut(self,x):
        mean = np.mean(x)
        std = np.std(x)
        x[x>(mean+std*2)] = (mean+std*2)
        x[x<(mean-std*2)] = (mean-std*2)
        return x
    def split_data(self,split):
        if(self.file[:3]=='com'):
            self.valX = self.data['valX'][:]
            self.valY = self.data['valY'][:][0]
            self.valY[self.valY>0] = 1
            self.val_parts = self.data['val_parts'][0].astype('int32')
            self.valdomY = [self.dom]*self.valY.shape[0]
            return
        taken = 0
        left = 0
        tmpX = None
        tmpY = None
        parts = []
        wav_name = []
        self.val_wav_name = []
        
        self.val_parts = []
        for j,x in enumerate(self.train_parts):
            if(taken<split*self.total):
                taken = taken + x
                if(self.valX is None):
                    if(self.seg):
                        self.valX = {k:self.trainX[k][:,left:x+left] for k in self.segments}
                    else:
                        self.valX = self.trainX[:,left:x+left]
                    self.valY = self.trainY[left:x+left]
                else:
                    if(self.seg):
                        self.valX = {k:np.concatenate((self.valX[k],self.trainX[k][:,left:x+left]),axis=1) for k in self.segments}
                    else:
                        self.valX = np.concatenate((self.valX,self.trainX[:,left:x+left]),axis=1)
                    self.valY = np.concatenate((self.valY,self.trainY[left:x+left]),axis=0)
                self.val_parts.append(x)
                self.val_wav_name.append(self.wav_name[j])
            else:
                if(tmpX is None):
                    if(self.seg):
                        tmpX = {k:self.trainX[k][:,left:x+left] for k in self.segments}
                    else:
                        tmpX = self.trainX[:,left:x+left]
                    tmpY = self.trainY[left:x+left]
                else:
                    if(self.seg):
                        tmpX = {k:np.concatenate((tmpX[k],self.trainX[k][:,left:x+left]),axis=1) for k in self.segments}
                    else:
                        tmpX = np.concatenate((tmpX,self.trainX[:,left:x+left]),axis=1)
                    tmpY = np.concatenate((tmpY,self.trainY[left:x+left]),axis=0)
                parts.append(x)
                wav_name.append(self.wav_name[j])
            left = left + x
        self.trainX = tmpX
        self.trainY = tmpY
        self.train_parts = parts
        self.wav_name = wav_name
        self.domainY = [self.dom]*self.trainY.shape[0]
        self.valdomY = [self.dom]*self.valY.shape[0]
        del tmpX,tmpY,parts
    def processCompare(self,severe):
        if(severe):
            print("fixed implementation, normal = 0, mild = 1, sever = 2. mild is being selected")
            self.sevX = None
            self.sevY = self.trainY[self.trainY%2==0]
            self.sev_parts = []
            self.sev_wav_name = []
            left = int(0)
            for j,x in enumerate(self.train_parts):
                x = int(x)
                if(all([i%2==0 for i in self.trainY[left:x+left]])):
                    if(self.sevX is None):
                        self.sevX = self.trainX[:,left:x+left]
                    else: 
                        self.sevX = np.concatenate((self.sevX,self.trainX[:,left:x+left]),axis=1)
                    self.sev_parts.append(x)
                    self.sev_wav_name.append(self.wav_name[j])
                left = x + left
                
            self.trainX = self.sevX
            self.trainY = self.sevY
            self.train_parts = self.sev_parts
            self.domainY = [self.dom]*self.trainY.shape[0]
            self.normal = Counter(self.trainY)[0]
            self.abnormal = Counter(self.trainY)[1]
            self.total = self.normal+self.abnormal
            self.normfiles, self.abnormfiles = self.parts()
        else:
            self.trainY[self.trainY>0] = 1
    def processE(self):
        self.trainY[self.trainY<0] = 0
        nn = ab = 0
        idx = []
        left = 0
        parts = []
        wav_name = []
        for j,x in enumerate(self.train_parts):
            if(all([(l == 0) for l in self.trainY[left:left+x]])):
                if(nn<self.abnormal):
                    idx = idx + [True]*x
                    nn = nn + x
                    parts.append(x)
                    wav_name.append(self.wav_name[j])
                else:idx = idx + [False]*x
            else:
                if(ab<self.abnormal):
                    idx = idx + [True]*x
                    ab = ab + x
                    parts.append(x)
                    wav_name.append(self.wav_name[j])
                else:idx = idx + [False]*x
            left = left + x
        self.trainY = self.trainY[idx]
        self.trainX = np.transpose(np.transpose(self.trainX)[idx])
        self.train_parts = parts
        self.wav_name = wav_name
        self.total = nn+ab
        self.domainY = self.domainY[:self.total]
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
    def details(self):
        if(self.trainX is  None):print("TrainX None/ ",end='')
        if(self.trainY is  None):print("trainY None/ ",end='')
        if(self.domainY is None):print("Dom Y/ ",end='')
        if(self.train_parts is  None):print("train parts/ ")
        if(self.valX is None):print("valX/ ",end='')
        if(self.valY is None) :print("valY/ ",end='')
        if(self.valdomY is None):print("Valdom/ ",end='')
        if(self.val_parts is  None):print("val parts/ ")