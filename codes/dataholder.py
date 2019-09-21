import matplotlib.pyplot as plt,h5py
from collections import Counter

class Data():
    def __init__(self,path,f,n):
        self.data = h5py.File(path+f, 'r')
        self.file = f
        self.dom = n
        self.trainX = self.data['trainX']
        self.trainY = self.data['trainY'][:][0]
        self.train_parts = self.data['train_parts'][0]
        self.valX = None
        self.valY = None
        self.normfiles, self.abnormfiles = self.parts()
        self.domainY = [n]*self.trainY.shape[0]
        if(f[:3]=='com'):
            self.trainY[self.trainY>0] = 1
            self.valX = self.data['valX']
            self.valY = self.data['valY'][:][0]
            self.valY[self.valY>0] = 1
        elif(f[:3]=='pas'):
            self.trainY[self.trainY<0] = 0
        else:
            self.trainY[self.trainY<0] = 0
        #print(Counter(self.trainY))
        self.normal = Counter(self.trainY)[0]
        self.abnormal = Counter(self.trainY)[1]
        self.total = self.normal+self.abnormal
        
    def parts(self):
        y = 0
        nn = 0
        ab = 0
        for x in self.train_parts:
            if(sum(self.trainY[y:y+int(x)])>0):
                ab = ab + 1
            else: nn = nn +1 
            y = int(x)+y
        return nn,ab
    def pie(self):
        colors = ['gold','lightskyblue']
        explode = (0.07, 0)
        size = [self.normal,self.abnormal]
        labels = ['N', 'Ab']
        plt.pie(size,labels=labels,colors = colors,startangle=140,explode=explode,
               shadow=True,autopct=self.value)
        plt.title(self.dom)